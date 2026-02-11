# Environment Variables Setup Guide

## Quick Setup

1. **Create or edit the `.env` file** in the `backend/` directory
2. **Add the following SMTP configuration:**

```env
# SMTP Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=recipient1@example.com,recipient2@example.com
```

## Detailed Configuration

### For Gmail Users

1. **Enable 2-Factor Authentication** on your Google account
2. **Generate an App Password:**
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and "Other (Custom name)"
   - Enter "MC4 Reports" as the name
   - Copy the generated 16-character password
3. **Add to `.env`:**
   ```env
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=your-email@gmail.com
   SMTP_PASSWORD=xxxx xxxx xxxx xxxx  # The app password (remove spaces)
   ALERT_EMAIL=recipient1@example.com,recipient2@example.com
   ```

### For Other Email Providers

#### Outlook/Hotmail
```env
SMTP_HOST=smtp-mail.outlook.com
SMTP_PORT=587
SMTP_USER=your-email@outlook.com
SMTP_PASSWORD=your-password
ALERT_EMAIL=recipient1@example.com,recipient2@example.com
```

#### Yahoo Mail
```env
SMTP_HOST=smtp.mail.yahoo.com
SMTP_PORT=587
SMTP_USER=your-email@yahoo.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=recipient1@example.com,recipient2@example.com
```

#### Custom SMTP Server
```env
SMTP_HOST=your-smtp-server.com
SMTP_PORT=587  # or 465 for SSL
SMTP_USER=your-email@domain.com
SMTP_PASSWORD=your-password
ALERT_EMAIL=recipient1@example.com,recipient2@example.com
```

## Alternative Variable Names

The code also supports these alternative variable names for compatibility:

- `MAIL_USERNAME` instead of `SMTP_USER`
- `MAIL_PASSWORD` instead of `SMTP_PASSWORD`
- `ALERT_EMAILS` instead of `ALERT_EMAIL`

## Complete .env Template

```env
# LLM Configuration (for Chatbot - optional)
LLM_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# SMTP Email Configuration (REQUIRED for Reports & Emails)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=recipient1@example.com,recipient2@example.com
```

## Testing

After configuring your `.env` file:

1. **Restart the backend server** (if it's running)
2. **Try sending an email** from the Reports & Emails page
3. **Check the backend logs** for any error messages

## Troubleshooting

### Error: "SMTP credentials not configured in .env"
- Make sure `.env` file exists in the `backend/` directory
- Check that `SMTP_USER` and `SMTP_PASSWORD` are set
- Restart the backend server after editing `.env`

### Error: "ALERT_EMAIL not configured in .env"
- Make sure `ALERT_EMAIL` or `ALERT_EMAILS` is set
- Use comma-separated list for multiple recipients
- Example: `ALERT_EMAIL=email1@example.com,email2@example.com`

### Gmail Authentication Errors
- Make sure you're using an **App Password**, not your regular password
- Verify 2-Factor Authentication is enabled
- Check that "Less secure app access" is not needed (use App Passwords instead)

### Connection Timeout
- Check your firewall settings
- Verify SMTP_HOST and SMTP_PORT are correct for your provider
- Try port 465 with SSL if 587 doesn't work
