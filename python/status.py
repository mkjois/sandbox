#!/usr/bin/env python
"""
Check for messages received in the past hours, specified as a command-line argument.
Gather and send to everyone.
If negative hours, send a reminder email to anyone who didn't send an update yet.
"""

EMAIL_ADDR = 'status@remeeting.com'
EMAIL_NAME = 'Remeeting Status'
PASSWORD = 'penthouse2150' # TODO: more secure?
RECIPIENTS = [('Arlo Faria', 'arlo@remeeting.com'),
              ('Korbinian Riedhammer', 'korbinian@remeeting.com'),
              ('Milo Cho', 'milo@remeeting.com'),
              ('Beryl Wang', 'beryl@remeeting.com'),
              ('Allen Guo', 'allen@remeeting.com'),
              ('Manny Jois', 'mjois@remeeting.com'),
              ('David Xie', 'x@remeeting.com')]

import sys
from imapclient import IMAPClient
import smtplib
import email
from email.mime.text import MIMEText
from email.utils import parsedate, parseaddr, formataddr
from time import mktime
from datetime import datetime, timedelta

# Optional arg: how many hours back to search for messages
# If negative, also send a reminder to those who didn't update
hours_back = 24
reminder = False
if len(sys.argv) > 1:
    hours_back = int(sys.argv[1])
if hours_back < 0:
    hours_back *= -1
    reminder = True
cutoff = datetime.now() - timedelta(hours=hours_back)

# Check email
gmail = IMAPClient('imap.gmail.com', ssl=True)
gmail.login(EMAIL_ADDR, PASSWORD)
gmail.select_folder('INBOX')
msgids = gmail.search('SINCE '+cutoff.strftime('%d-%b-%Y'))
fetched = gmail.fetch(msgids, ['RFC822']).values()
gmail.logout()

# Parse emails
updates = {}
for msg in [email.message_from_string(m['RFC822']) for m in fetched]:
    if cutoff > datetime.fromtimestamp(mktime(parsedate(msg['Date']))):
        continue

    # Sort by real name, in case a person uses multiple addresses
    realname, email_addr = parseaddr(msg['From'])

    # Get the message body, or the subject line
    body = None
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            body = part.get_payload()
    if not body or not body.strip():
        body = msg['Subject']

    for line in body.strip().split('\n'):
        if line.strip():
            if realname in updates:
                updates[realname].append(line.strip())
            else:
                updates[realname] = [line.strip()]

# Send reminders or updates
smtp = smtplib.SMTP('smtp.gmail.com:587')
smtp.starttls()
smtp.login(EMAIL_ADDR, PASSWORD)
if reminder:
    for realname, email_addr in [r for r in RECIPIENTS if r[0] not in updates.keys()]:
        msg = MIMEText("Please reply to this message (without quoting this text!) or write to %s\n" % EMAIL_ADDR)
        msg['From'] = formataddr((EMAIL_NAME, EMAIL_ADDR))
        msg['To'] = formataddr((realname, email_addr))
        msg['Subject'] = 'Share your progress since ' + cutoff.strftime('%A, %B %d')
        smtp.sendmail(EMAIL_ADDR, email_addr, msg.as_string())
else:
    body = ''
    for who in sorted(updates.keys()):
        body += who + ':\n'
        for what in updates[who]:
            body += ' - ' + what + '\n'
    for realname, email_addr in RECIPIENTS:
        msg = MIMEText(body)
        msg['From'] = formataddr((EMAIL_NAME, EMAIL_ADDR))
        msg['To'] = formataddr((realname, email_addr))
        msg['Subject'] = 'Progress as of ' + datetime.now().strftime('%A, %B %d')
        smtp.sendmail(EMAIL_ADDR, email_addr, msg.as_string())
smtp.quit()
