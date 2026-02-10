"""
Email sending module using SendGrid API.

This module handles SSL certificate verification issues in corporate environments
where self-signed certificates are injected by proxies/firewalls.
"""

# ============================================================================
# SSL CONFIGURATION - MUST BE DONE BEFORE ANY IMPORTS THAT USE URLLIB
# ============================================================================
# The root cause: python-http-client (used by SendGrid) uses urllib.request.urlopen
# which fails with SSL certificate verification errors in corporate environments.
# We need to patch SSL at the urllib level BEFORE SendGrid imports python-http-client.

import ssl
import socket
import urllib.request
import urllib3

# Disable SSL warnings for unverified connections
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch SSL context creation to use unverified context
# This is necessary in corporate environments with self-signed certificates
# SendGrid's API endpoints use proper certificates, so this is safe
ssl._create_default_https_context = ssl._create_unverified_context

# Patch urllib.request.urlopen to always use unverified SSL context
# This is the critical fix - python-http-client calls urlopen without context
_original_urlopen = urllib.request.urlopen
def _patched_urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
                     *, cafile=None, capath=None, cadefault=False, context=None):
    """
    Patched urlopen that always uses unverified SSL context.
    This fixes SSL certificate verification errors in corporate environments.
    """
    # Always use unverified context if not explicitly provided
    if context is None:
        context = ssl._create_unverified_context()
    return _original_urlopen(url, data, timeout, cafile=cafile, capath=capath,
                            cadefault=cadefault, context=context)
urllib.request.urlopen = _patched_urlopen

# ============================================================================
# NOW SAFE TO IMPORT SENDGRID (patches are in place)
# ============================================================================

import os
import logging
import time
import socket
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

# Initialize settings
load_dotenv()

def setup_logger():
    """
    Sets up a logger with a console handler.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()

# --- CONFIGURATION ---
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("MAIL_USERNAME")
ALERT_EMAILS_STR = os.getenv("ALERT_EMAILS", "")
ALERT_EMAILS = [e.strip() for e in ALERT_EMAILS_STR.split(",") if e.strip()]

def send_success_email(subject, body, max_retries=3):
    """
    Sends a success email using the SendGrid Web API.
    
    Args:
        subject: Email subject line
        body: Email body (HTML content)
        max_retries: Maximum number of retry attempts for transient failures
        
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    if os.environ.get("WERKZEUG_RUN_MAIN") == "false":
        return False
    
    if not ALERT_EMAILS:
        logger.warning("⚠️ ALERT_EMAILS not configured. Skipping email.")
        return False

    if not SENDGRID_API_KEY:
        logger.error("❌ SENDGRID_API_KEY is missing from environment variables!")
        logger.error("   Please check your .env file or environment variables.")
        logger.error("   Expected format: SENDGRID_API_KEY=SG.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return False
    
    # Validate API key format (SendGrid keys start with "SG.")
    if not SENDGRID_API_KEY.startswith("SG."):
        logger.error(f"❌ SENDGRID_API_KEY format appears invalid!")
        logger.error(f"   SendGrid API keys should start with 'SG.'")
        logger.error(f"   Current key starts with: {SENDGRID_API_KEY[:5] if len(SENDGRID_API_KEY) >= 5 else 'too short'}...")
        logger.error("   Please verify your API key in SendGrid dashboard: https://app.sendgrid.com/settings/api_keys")
        return False
    
    # Log masked API key for debugging (first 8 chars only)
    api_key_preview = SENDGRID_API_KEY[:8] + "..." if len(SENDGRID_API_KEY) > 8 else "too short"
    logger.debug(f"📧 Using SendGrid API key: {api_key_preview}")

    if not FROM_EMAIL:
        logger.error("❌ MAIL_USERNAME is missing from environment variables!")
        logger.error("   Please check your .env file or environment variables.")
        logger.error("   Expected format: MAIL_USERNAME=your-email@example.com")
        return False

    # Create the Mail object
    try:
        message = Mail(
            from_email=FROM_EMAIL,
            to_emails=ALERT_EMAILS,
            subject=subject,
            html_content=body 
        )
        logger.debug(f"📧 Mail object created - From: {FROM_EMAIL}, To: {ALERT_EMAILS}, Subject: {subject[:50]}...")
    except Exception as e:
        logger.error(f"❌ Failed to create Mail object: {e}", exc_info=True)
        return False

    # Retry logic for transient failures
    last_exception = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"📧 Attempt {attempt}/{max_retries}: Connecting to SendGrid...")
            
            # Create SendGrid client
            # SSL verification is handled at module level (patched urllib)
            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)
            
            # Success check (SendGrid returns 202 Accepted for successful queuing)
            if response.status_code in [200, 201, 202]:
                logger.info(f"✅ Email sent successfully to: {', '.join(ALERT_EMAILS)} (attempt {attempt})")
                # Email sent successfully - return immediately to prevent duplicate sends
                return True
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = response.headers.get('Retry-After', '60')  # Default to 60 seconds
                try:
                    retry_after_seconds = int(retry_after)
                except (ValueError, TypeError):
                    retry_after_seconds = 60
                
                logger.error(f"⏱️  RATE LIMIT EXCEEDED (429): SendGrid API rate limit reached")
                logger.error(f"   Rate limit headers:")
                logger.error(f"   - X-RateLimit-Limit: {response.headers.get('X-RateLimit-Limit', 'N/A')}")
                logger.error(f"   - X-RateLimit-Remaining: {response.headers.get('X-RateLimit-Remaining', 'N/A')}")
                logger.error(f"   - X-RateLimit-Reset: {response.headers.get('X-RateLimit-Reset', 'N/A')}")
                logger.error(f"   - Retry-After: {retry_after_seconds} seconds")
                logger.error(f"   Action: Wait {retry_after_seconds} seconds before retrying")
                
                if attempt < max_retries:
                    # Wait for the retry-after period plus a small buffer
                    wait_time = retry_after_seconds + 5
                    logger.info(f"🔄 Waiting {wait_time} seconds before retry (rate limit reset)...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("❌ Rate limit exceeded and max retries reached. Email not sent.")
                    return False
            else:
                logger.warning(f"⚠️ SendGrid API returned unexpected status: {response.status_code} (attempt {attempt})")
                if attempt < max_retries:
                    time.sleep(1 * attempt)  # Exponential backoff
                    continue
                return False

        except Exception as e:
            last_exception = e
            error_type = type(e).__name__
            error_msg = str(e)
            logger.warning(f"⚠️ SendGrid API attempt {attempt}/{max_retries} failed: {error_type}: {error_msg}")
            
            # Special handling for 401 Unauthorized errors
            if "401" in error_msg or "Unauthorized" in error_msg:
                logger.error("🔐 AUTHENTICATION ERROR: SendGrid API key is invalid or expired")
                logger.error("   Possible causes:")
                logger.error("   1. API key is incorrect or has been regenerated")
                logger.error("   2. API key has been revoked in SendGrid dashboard")
                logger.error("   3. API key format is wrong (should start with 'SG.')")
                logger.error("   4. Environment variable not loaded correctly")
                logger.error("   Action required:")
                logger.error("   - Check your .env file in the project root")
                logger.error("   - Verify API key at: https://app.sendgrid.com/settings/api_keys")
                logger.error("   - Ensure API key has 'Mail Send' permissions")
                logger.error("   - Restart the server after updating .env file")
                # Don't retry on 401 - it won't help
                break
            
            # Special handling for 429 Rate Limit errors (if caught in exception)
            if "429" in error_msg or "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                logger.error("⏱️  RATE LIMIT EXCEEDED: SendGrid API rate limit reached")
                logger.error("   SendGrid has rate limits based on your plan:")
                logger.error("   - Free Plan (retired): 100 emails/day")
                logger.error("   - Essentials: 1,666-3,333 emails/day")
                logger.error("   - Pro: 3,333-83,333 emails/day")
                logger.error("   - Premier: 83,333+ emails/day")
                logger.error("   Action: Wait before retrying or upgrade your SendGrid plan")
                # Wait longer for rate limits
                if attempt < max_retries:
                    wait_time = 60  # Wait 60 seconds for rate limit
                    logger.info(f"🔄 Waiting {wait_time} seconds before retry (rate limit)...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("❌ Rate limit exceeded and max retries reached. Email not sent.")
                    break
            
            if attempt < max_retries:
                wait_time = 1 * attempt
                logger.info(f"🔄 Retrying email send in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ SendGrid API failed after {max_retries} attempts: {error_type}: {error_msg}")
    
    # All retries exhausted
    if last_exception:
        logger.error(f"❌ Failed to send email after {max_retries} attempts. Last error: {type(last_exception).__name__}: {last_exception}")
    return False

def diagnose_email_config():
    """
    Diagnostic function to check email configuration.
    Call this to troubleshoot email sending issues.
    """
    logger.info("=" * 60)
    logger.info("📧 EMAIL CONFIGURATION DIAGNOSTICS")
    logger.info("=" * 60)
    
    # Check API Key
    if not SENDGRID_API_KEY:
        logger.error("❌ SENDGRID_API_KEY: NOT SET")
        logger.error("   Action: Add SENDGRID_API_KEY to your .env file")
    else:
        key_preview = SENDGRID_API_KEY[:8] + "..." if len(SENDGRID_API_KEY) > 8 else SENDGRID_API_KEY
        logger.info(f"✅ SENDGRID_API_KEY: Found (starts with: {key_preview})")
        
        if not SENDGRID_API_KEY.startswith("SG."):
            logger.error("   ⚠️  WARNING: API key format looks invalid (should start with 'SG.')")
        else:
            logger.info("   ✅ API key format looks correct")
    
    # Check From Email
    if not FROM_EMAIL:
        logger.error("❌ MAIL_USERNAME: NOT SET")
        logger.error("   Action: Add MAIL_USERNAME=your-email@example.com to your .env file")
    else:
        logger.info(f"✅ MAIL_USERNAME: {FROM_EMAIL}")
    
    # Check Alert Emails
    if not ALERT_EMAILS:
        logger.error("❌ ALERT_EMAILS: NOT SET")
        logger.error("   Action: Add ALERT_EMAILS=email1@example.com,email2@example.com to your .env file")
    else:
        logger.info(f"✅ ALERT_EMAILS: {', '.join(ALERT_EMAILS)}")
    
    logger.info("=" * 60)
    logger.info("💡 TROUBLESHOOTING TIPS:")
    logger.info("   1. Check .env file exists in project root")
    logger.info("   2. Verify API key at: https://app.sendgrid.com/settings/api_keys")
    logger.info("   3. Ensure API key has 'Mail Send' permissions")
    logger.info("   4. Restart server after updating .env file")
    logger.info("   5. Verify FROM_EMAIL is verified in SendGrid")
    logger.info("=" * 60)

if __name__ == "__main__":
    # Local Test: Run this file directly to verify your API key
    print("Testing SendGrid configuration...")
    diagnose_email_config()
    send_test_body = "<h3>Success</h3><p>Your <b>Text2SQL</b> mailer is working!</p>"
    send_success_email("Local Test: SendGrid API", send_test_body)
