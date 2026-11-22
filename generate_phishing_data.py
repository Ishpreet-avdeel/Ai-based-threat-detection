"""
generate_phishing_data.py
-------------------------
Generates synthetic email datasets for the phishing detection module.

Creates two files inside data/:

1. data/phishing_train.csv  (labeled, ~400 rows, balanced)
       email_text : subject + body combined into one string
       label      : 1 = Phishing, 0 = Legitimate
   200 phishing emails (urgency, fake invoice, password reset, CEO wire transfer,
   account suspension, gift card, fake shared document ...) and 200 legitimate
   workplace emails (meetings, reports, HR updates, project status ...).

2. data/sample_emails.csv  (unlabeled, 50 rows, for the dashboard demo)
       timestamp, sender, subject, body, recipient
   ~42 safe workplace emails and 8 tricky phishing emails. There is NO label
   column, exactly like a real inbox export.

Usage:
    python generate_phishing_data.py
"""

import random
import re
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_FILE = DATA_DIR / "phishing_train.csv"
SAMPLE_FILE = DATA_DIR / "sample_emails.csv"

TRAIN_PER_CLASS = 200          # 200 phishing + 200 legitimate
SAMPLE_SAFE_COUNT = 42         # safe emails in the demo file
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
COMPANY_DOMAIN = "acme-corp.com"

# --------------------------------------------------------------------------- #
# Shared vocabulary for slot filling
# --------------------------------------------------------------------------- #
FIRST_NAMES = ["Alex", "Priya", "Daniel", "Maria", "Kenji", "Sara", "Omar", "Elena",
               "Chris", "Fatima", "Liam", "Nina", "Rahul", "Grace", "Tom", "Hannah"]
LAST_NAMES = ["Patel", "Nguyen", "Garcia", "Smith", "Kim", "Rossi", "Khan", "Brown",
              "Silva", "Cohen", "Walker", "Ivanov", "Tanaka", "Lopez", "Evans", "Ahmed"]
BRANDS = ["PayPal", "Microsoft 365", "Amazon", "Netflix", "Chase Bank", "Apple ID",
          "DocuSign", "Google Workspace", "LinkedIn", "Dropbox", "DHL", "Zoom",
          "Wells Fargo", "Adobe"]
EXECS = ["Michael Grant", "Susan Clarke", "David Walsh", "Laura Bennett"]
BAD_TLDS = ["xyz", "top", "info", "click", "site", "online", "net"]
URL_WORDS = ["secure", "verify", "login", "update", "account", "confirm", "auth", "signin", "support"]
URL_PATHS = ["login", "verify", "secure", "update", "confirm", "signin"]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["9:30 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:30 PM", "4:00 PM"]
PROJECTS = ["Atlas", "Beacon", "Orion", "Helios", "Nimbus", "Phoenix", "Aurora", "Mercury"]
DEPTS = ["Finance", "Engineering", "Marketing", "Operations", "Sales", "Support", "Legal"]
MONTHS = ["January", "March", "April", "June", "July", "September", "October", "December"]
ROOMS = ["Conference Room B", "the third floor meeting room", "Room 204", "the main boardroom"]


# --------------------------------------------------------------------------- #
# Phishing templates
# --------------------------------------------------------------------------- #
PHISHING_CATEGORIES = [
    {   # account suspension / urgency
        "subjects": [
            "URGENT: Your {brand} account has been suspended",
            "Action required: unusual sign-in detected on your {brand} account",
            "Final warning: your {brand} account will be closed in {hours} hours",
            "Security alert: verify your {brand} account now",
            "Your {brand} account is locked - restore access immediately",
        ],
        "greetings": ["Dear customer,", "Dear user,", "Attention,", "Hello,", "Dear {brand} member,"],
        "bodies": [
            "We detected suspicious activity and your account has been temporarily suspended. "
            "You must verify your identity within {hours} hours or your account will be permanently "
            "deleted. Click here to restore access: {url}",
            "Your account was accessed from an unrecognized device. For your protection we have limited "
            "some features. Confirm your login details immediately to avoid permanent suspension: {url}",
            "Our security system flagged multiple failed sign-in attempts. Failure to verify your "
            "information within {hours} hours will result in account closure. Verify now: {url}",
        ],
        "closers": ["Regards, {brand} Security Team",
                    "Thank you for your immediate attention. {brand} Account Services",
                    "Do not ignore this message. {brand} Support"],
    },
    {   # fake invoice / payment
        "subjects": [
            "Invoice {inv} overdue - immediate payment required",
            "Payment failed for invoice {inv}",
            "Your {brand} subscription invoice for ${amount}",
            "Final notice: unpaid invoice {inv}",
            "Outstanding balance of ${amount} - pay today",
        ],
        "greetings": ["Dear customer,", "Hello,", "Dear accounts payable,", "Attention,"],
        "bodies": [
            "Please find the attached invoice for ${amount}. Payment is overdue and a late fee will "
            "apply. Open the attachment or click {url} to review and pay immediately to avoid service "
            "interruption.",
            "We were unable to process your last payment of ${amount}. Update your billing information "
            "within {hours} hours to prevent cancellation: {url}",
            "Your invoice {inv} is past due. Download the attached payment form, enter your bank details "
            "and submit it today. Legal action may follow if payment is not received.",
        ],
        "closers": ["Billing Department", "Accounts Receivable, {brand}", "Regards, {brand} Billing Team"],
    },
    {   # password reset / expiry
        "subjects": [
            "Password expires today - reset now",
            "Reset your {brand} password immediately",
            "Your mailbox password will expire in {hours} hours",
            "Confirm your password to keep your account active",
            "IT notice: mandatory password update required",
        ],
        "greetings": ["Dear user,", "Hello,", "Dear employee,", "Attention,"],
        "bodies": [
            "Your password will expire in {hours} hours. Confirm your current password and click the "
            "link below to keep your mailbox active: {url} Failure to act will lock your account.",
            "We received a request to reset your password. If you did not make this request, sign in now "
            "and verify your credentials at {url} to secure your account.",
            "As part of a mandatory security upgrade, all users must re-enter their username and password "
            "at {url} before the deadline. Accounts that are not updated will be disabled.",
        ],
        "closers": ["IT Helpdesk", "Security Team, {brand}", "System Administrator"],
    },
    {   # CEO wire-transfer fraud
        "subjects": [
            "Quick favor - urgent",
            "Confidential: wire transfer needed today",
            "Need your help with a payment",
            "Are you available? Urgent request",
            "Urgent payment to new vendor - keep confidential",
        ],
        "greetings": ["Hi {name},", "{name},", "Hello {name},"],
        "bodies": [
            "I am in a meeting and cannot take calls right now. I need you to process an urgent wire "
            "transfer of ${amount} to a new vendor today. Keep this confidential and reply as soon as it "
            "is done. I will send the account details by email.",
            "Are you at your desk? I need a favor that must stay between us. Please arrange an immediate "
            "bank transfer of ${amount} for a confidential acquisition and send me the confirmation. "
            "Do not discuss this with anyone.",
            "I am traveling and unable to approve this through the usual channels. Please wire ${amount} "
            "to the account I will send you and reply with the receipt. This is urgent and time sensitive.",
        ],
        "closers": ["Thanks, {exec}, CEO", "Regards, {exec}", "Sent from my iPhone - {exec}"],
    },
    {   # prize / gift card / reward
        "subjects": [
            "Congratulations! You have won a ${amount} gift card",
            "Claim your reward now - limited time",
            "You have been selected for an exclusive {brand} reward",
            "Survey reward: claim your free gift today",
        ],
        "greetings": ["Dear winner,", "Congratulations,", "Dear customer,", "Hello lucky user,"],
        "bodies": [
            "You have been randomly selected to receive a ${amount} {brand} gift card. Claim your prize "
            "within {hours} hours by confirming your details at {url} or it will expire.",
            "Complete this short survey and receive a free reward. Enter your name, address and card "
            "number to cover the small shipping fee: {url}",
            "Your account qualifies for a special cash reward. Click the link below now to claim it "
            "before the offer ends: {url}",
        ],
        "closers": ["The Rewards Team", "{brand} Promotions", "Offer expires soon!"],
    },
    {   # fake shared document / e-signature
        "subjects": [
            "{name} shared a document with you",
            "Please review and sign: payroll update",
            "Secure document waiting for your signature",
            "New voice message and shared file",
            "Re: updated contract - view online",
        ],
        "greetings": ["Hello,", "Hi,", "Dear colleague,"],
        "bodies": [
            "{name} has shared a confidential file with you. Sign in with your email and password to view "
            "the secure document: {url}",
            "A document requires your signature today. Open the secure link and log in with your corporate "
            "credentials to review and sign: {url}",
            "You have received a new protected file. Click the link below and enter your email password "
            "to download the attachment: {url}",
        ],
        "closers": ["{brand} Secure Share", "Sent via {brand}", "Do not forward this email."],
    },
    {   # delivery / customs fee
        "subjects": [
            "Delivery attempt failed - action required",
            "Your parcel is on hold: pay redelivery fee",
            "Package could not be delivered",
            "Customs fee required for your shipment",
        ],
        "greetings": ["Dear customer,", "Hello,", "Attention,"],
        "bodies": [
            "We were unable to deliver your package. Pay a small redelivery fee of ${small} at {url} or "
            "your parcel will be returned to the sender within {hours} hours.",
            "Your shipment is held at customs. Confirm your address and pay the outstanding fee of "
            "${small} using the link below: {url}",
        ],
        "closers": ["{brand} Delivery Service", "Customer Service, {brand}"],
    },
    {   # tax refund / bank alert
        "subjects": [
            "You are eligible for a tax refund of ${amount}",
            "Bank alert: confirm your card details",
            "Suspicious transaction on your account - verify now",
            "Your refund is pending - claim today",
        ],
        "greetings": ["Dear taxpayer,", "Dear customer,", "Hello,"],
        "bodies": [
            "After the latest calculation of your fiscal activity we determined you are eligible for a "
            "refund of ${amount}. Submit your bank details and social security number at {url} to "
            "receive payment.",
            "We noticed an unauthorized transaction of ${amount} on your card. Confirm your card number, "
            "PIN and security code immediately to cancel it: {url}",
        ],
        "closers": ["Tax Refund Department", "{brand} Fraud Prevention", "Regards, {brand} Security"],
    },
    {   # fake HR / payroll
        "subjects": [
            "Updated salary information - login required",
            "Mandatory HR policy acknowledgement - sign in now",
            "Action required: confirm your direct deposit details",
            "Bonus payment pending - verify your account",
        ],
        "greetings": ["Dear employee,", "Hello,", "Hi {name},"],
        "bodies": [
            "HR is updating payroll records. Log in with your employee credentials at {url} and confirm "
            "your bank details by end of day to avoid payment delays.",
            "All staff must acknowledge the new compensation policy today. Sign in at {url} using your "
            "work email and password to view and accept it.",
        ],
        "closers": ["Human Resources Department", "Payroll Team", "HR Services"],
    },
    {   # mailbox full / account deactivation
        "subjects": [
            "Your mailbox is almost full",
            "Email storage limit exceeded - action required",
            "Incoming messages are being blocked",
            "Webmail account deactivation notice",
        ],
        "greetings": ["Dear user,", "Hello,", "Attention,"],
        "bodies": [
            "Your mailbox has reached its storage limit and incoming messages will be rejected. Click "
            "here to upgrade your storage and verify your account within {hours} hours: {url}",
            "Your webmail account will be deactivated unless you re-validate it now. Login to confirm "
            "your account ownership: {url}",
        ],
        "closers": ["Webmail Administrator", "Email Support Team", "IT Support"],
    },
]

# --------------------------------------------------------------------------- #
# Legitimate workplace templates (used for the training set)
# --------------------------------------------------------------------------- #
GENERIC_LEGIT_CLOSERS = ["Thanks, {sender}", "Best regards, {sender}", "Cheers, {sender}",
                         "Regards, {sender}", "Thank you, {sender}"]

LEGIT_CATEGORIES = [
    {   # meetings
        "subjects": [
            "Team meeting moved to {day} at {time}",
            "Agenda for {project} sync",
            "Reminder: {project} planning meeting on {day}",
            "Can we reschedule our 1:1?",
            "Meeting notes from {day}",
        ],
        "greetings": ["Hi team,", "Hello everyone,", "Hi {name},", "Good morning,"],
        "bodies": [
            "Please note the weekly meeting has been moved to {day} at {time} in {room}. The agenda is "
            "attached and we will cover the {project} timeline and open action items. Let me know if you "
            "have a conflict.",
            "Thanks for joining today. I have attached the notes from our {project} discussion. Please "
            "review the action items assigned to you and share any corrections by {day}.",
            "I would like to reschedule our one on one to {day} at {time}. Does that work for you? Happy "
            "to move it if you are busy.",
        ],
    },
    {   # reports
        "subjects": [
            "Q{quarter} sales report attached",
            "Weekly status report - {project}",
            "Monthly budget summary for {dept}",
            "Draft report for review",
            "Updated project dashboard - {project}",
        ],
        "greetings": ["Hi all,", "Hello,", "Hi {name},"],
        "bodies": [
            "Please find attached the {dept} report for this period. Revenue is in line with forecast and "
            "the main variances are explained on page {num}. I would appreciate your comments before {day}.",
            "Here is the weekly status update for {project}. We completed {num} tasks this week, two items "
            "are in review and the next milestone is on track. No blockers at the moment.",
            "The draft report is ready for your feedback. Highlights include improved delivery times and a "
            "small budget underspend. Let me know if you want any changes before it goes to leadership.",
        ],
    },
    {   # HR updates
        "subjects": [
            "Updated holiday calendar for 2026",
            "Benefits enrollment window opens in {month}",
            "Welcome {name} to the {dept} team",
            "Reminder: submit your timesheet by Friday",
            "Annual performance review schedule",
            "Office closed for the public holiday",
        ],
        "greetings": ["Hello everyone,", "Hi team,", "Dear colleagues,"],
        "bodies": [
            "Please see the updated holiday calendar attached. The office will be closed on the public "
            "holidays listed and normal hours resume the following working day. Reach out to HR if you "
            "have any questions.",
            "We are pleased to welcome {name} who joins the {dept} team on Monday. Please say hello and "
            "help them settle in. A short introduction lunch is planned for {day}.",
            "This is a friendly reminder to submit your timesheet by Friday afternoon. If you will be on "
            "leave, please update your calendar so your manager can plan coverage.",
            "The annual benefits enrollment window opens in {month}. You will receive details through the "
            "usual HR channels and can attend an optional information session in {room}.",
        ],
        "closers": ["Thanks, Human Resources", "Best regards, HR Team", "Regards, People Operations"],
    },
    {   # project status
        "subjects": [
            "Project {project} - milestone update",
            "Sprint review notes",
            "Release candidate ready for QA",
            "Code review needed for {project}",
            "Deployment completed successfully",
        ],
        "greetings": ["Hi team,", "Hello,", "Hi {name},"],
        "bodies": [
            "The {project} release candidate is ready for QA. The main features are merged and the "
            "regression tests are passing. Please test your areas and log any issues in the tracker by {day}.",
            "Quick update on {project}: the migration finished without errors, performance is stable and "
            "the next sprint starts on {day}. Thanks everyone for the hard work.",
            "Could you review my pull request when you have a moment? It touches the reporting module and "
            "adds unit tests. No rush, anytime before {day} works.",
        ],
    },
    {   # social / office life
        "subjects": [
            "Team lunch on {day}",
            "Farewell cake for {name}",
            "Office social event - RSVP",
            "Coffee catch up this week?",
            "Happy work anniversary {name}",
        ],
        "greetings": ["Hi all,", "Hello team,", "Hi {name},"],
        "bodies": [
            "We are planning a team lunch on {day} at noon. Let me know your dietary preferences so I can "
            "book the table. Everyone is welcome.",
            "{name} is leaving at the end of the month and we will have cake in the kitchen on {day} "
            "afternoon. Please drop by and say goodbye if you can.",
            "Are you free for a coffee this week? It would be great to catch up and hear how the new "
            "project is going. Any day after lunch works for me.",
        ],
    },
    {   # genuine IT notices (contain words like password / security on purpose)
        "subjects": [
            "Scheduled maintenance this Saturday night",
            "Reminder: complete the security awareness training",
            "New laptop refresh schedule",
            "Password policy update - no action needed",
            "VPN maintenance window on {day}",
        ],
        "greetings": ["Hello everyone,", "Hi all,", "Dear colleagues,"],
        "bodies": [
            "The IT team will perform scheduled maintenance on the file servers this Saturday from 10 PM "
            "to 2 AM. Some systems may be briefly unavailable. No action is needed from you.",
            "This is a reminder to complete the annual security awareness training on the intranet by the "
            "end of the month. The course takes about 30 minutes and you can pause and resume at any time.",
            "As announced in the staff meeting, our password policy now recommends longer passphrases. No "
            "action is needed today. Visit the intranet security page if you would like tips on choosing "
            "a good passphrase.",
            "Laptop refresh for the {dept} department starts in {month}. Please back up your files to the "
            "shared drive and keep an eye on your calendar for your appointment with the IT desk.",
        ],
        "closers": ["Thanks, IT Team", "Best regards, IT Service Desk", "Regards, Infrastructure Team"],
    },
    {   # genuine finance notices (contain words like invoice / payment on purpose)
        "subjects": [
            "Invoice {inv} approved and processed",
            "Expense report approved",
            "Payment received for invoice {inv}",
            "Purchase order for {dept} supplies",
            "Budget approval request for {project}",
        ],
        "greetings": ["Hi {name},", "Hello,", "Hi team,"],
        "bodies": [
            "Your expense report has been approved and will be paid with the next payroll run. No further "
            "action is required. Contact finance if anything looks incorrect.",
            "We have received payment for invoice {inv}. Thank you for the prompt settlement. The receipt "
            "is attached for your records.",
            "I have approved the purchase order for {dept} supplies. Please coordinate delivery with the "
            "operations team and send the signed delivery note to accounts when it arrives.",
        ],
    },
    {   # client communication
        "subjects": [
            "Following up on our proposal",
            "Contract review - comments attached",
            "Thanks for the call today",
            "Feedback on the {project} demo",
            "Quarterly business review with the client",
        ],
        "greetings": ["Hi {name},", "Hello,", "Good afternoon,"],
        "bodies": [
            "Thank you for taking the time to speak today. As discussed, I have attached our proposal and "
            "a draft timeline. Happy to walk through any questions on a call next week.",
            "Please find our comments on the contract attached. Most clauses are fine and we have suggested "
            "small changes to the delivery schedule. Let me know when legal has had a chance to review.",
            "The client was very happy with the {project} demo and has asked for a few small tweaks. I will "
            "share the list of requested changes tomorrow so we can plan the next iteration.",
        ],
    },
]

# --------------------------------------------------------------------------- #
# Templates used ONLY for the demo file (different wording from training data)
# --------------------------------------------------------------------------- #
SAFE_SAMPLE_TEMPLATES = [
    ("Lunch & learn on cloud costs",
     "Hi all, a lunch and learn on cloud cost optimisation is scheduled for {day} at {time} in {room}. "
     "Pizza will be provided. Please RSVP so we can order enough food."),
    ("Slides from yesterday's workshop",
     "Hi {name}, thanks for attending the workshop. The slides and recording are saved in the shared "
     "folder. Feel free to send questions my way."),
    ("Quick question about the {project} timeline",
     "Hello, could you confirm whether the {project} deliverables are still due on {day}? I want to "
     "update the plan before our next stand-up."),
    ("Team offsite ideas",
     "Hi team, we are collecting ideas for the next offsite. Please add your suggestions to the shared "
     "document by {day}. Budget and dates will be decided afterwards."),
    ("Vacation request approved",
     "Hi {name}, your leave request has been approved. Enjoy the time off and please add an out of "
     "office message to your mailbox before you go."),
    ("Interview schedule for the {dept} role",
     "Hello, the interviews for the open {dept} position are scheduled for {day}. Please check your "
     "calendar invite and read the candidate resumes beforehand."),
    ("Customer feedback summary",
     "The latest customer survey results are in. Satisfaction increased slightly and support response "
     "time improved. A full summary is in the shared drive for anyone interested."),
    ("Parking and building access update",
     "Please note that the north car park will be closed for resurfacing next week. Use the east "
     "entrance and bring your badge. Sorry for the inconvenience."),
    ("Re: {project} requirements",
     "Thanks for the details, {name}. I have updated the requirements document with your comments and "
     "will walk the team through it at the {day} meeting."),
    ("Invoice {inv} approved",
     "Hello, your invoice {inv} has been approved by finance and will be paid within the standard "
     "terms. No action is required on your side."),
    ("Reminder: complete the compliance training",
     "This is a reminder to finish the annual compliance course on the learning portal by the end of the "
     "month. It takes around 45 minutes and can be completed in several sessions."),
    ("Scheduled system maintenance on {day}",
     "IT will carry out routine maintenance on {day} evening. Email and file shares may be unavailable "
     "for about an hour. You do not need to do anything."),
    ("Welcome to the team, {name}!",
     "Please join me in welcoming {name} to the {dept} department. {name} starts today and will be "
     "sitting near the kitchen. Let us make them feel at home."),
    ("Updated org chart",
     "The updated organisation chart is attached. It reflects the recent changes in {dept} and the new "
     "reporting lines effective next month."),
    ("Thank you for your help",
     "Hi {name}, I just wanted to say thanks for covering the support queue last week. It made a big "
     "difference and the client noticed. Coffee is on me."),
    ("Budget forecast for next quarter",
     "I have attached the forecast for next quarter. Please review your cost centre and flag any items "
     "that look off by {day} so finance can finalise the numbers."),
    ("Design review feedback",
     "Thanks for sharing the mockups. The layout looks great overall. I left a few comments on the "
     "navigation and the colour contrast in the shared file."),
    ("Bug report: export button not working",
     "Hi dev team, a user reported that the export button on the reports page fails with a timeout. "
     "Steps to reproduce are in the ticket. Could someone take a look today?"),
    ("Meeting minutes - {project} steering committee",
     "The minutes from the steering committee are attached. Key decisions include extending the pilot "
     "by two weeks and hiring one additional analyst."),
    ("Training room booking",
     "I have booked {room} for the onboarding session on {day} from {time}. Please bring a laptop and "
     "your notes from the previous module."),
    ("Weekly newsletter: what's new at the company",
     "This week's newsletter covers our new partnership, the upcoming charity run and a profile of the "
     "{dept} team. Enjoy the read and send us your feedback."),
    ("Expense policy reminder",
     "A quick reminder that expense claims must be submitted within 30 days with receipts attached. "
     "Reach out to finance if you need help with the new claims system."),
    ("Can you review my draft?",
     "Hi {name}, I have written a first draft of the {project} proposal. Could you take a look this "
     "week and tell me what is missing? I appreciate your input."),
    ("Server migration completed",
     "The migration of the internal wiki to the new servers completed successfully overnight. Everything "
     "is working normally. Please report anything unusual to the helpdesk."),
    ("Password manager rollout",
     "We are rolling out the company password manager to all departments in {month}. A short guide and "
     "optional drop in sessions will be available. Nothing is required from you today."),
    ("Shipment of office supplies arrived",
     "The new monitors and keyboards were delivered to the {dept} floor. Please collect yours from "
     "reception and sign the delivery sheet when you pick them up."),
]

# Eight hand-written "tricky" phishing emails: lookalike domains, polite tone,
# no obvious shouting. These are NOT copies of any training email.
TRICKY_PHISHING = [
    ("it-support@acme-corp-helpdesk.com", "Mailbox storage quota almost reached",
     "Hello, your mailbox has reached 98% of its quota and new messages may be delayed. To avoid losing "
     "incoming mail, please re-validate your account within 24 hours at "
     "http://acme-mail-quota.top/validate using your normal email login."),
    ("michael.grant.ceo@gmail.com", "Are you available?",
     "Hi Priya, are you at your desk? I need a small favor that must stay between us for now. I am "
     "traveling and need gift cards purchased for a client today. I will reimburse you later. Reply as "
     "soon as you see this."),
    ("no-reply@docusign-secure-view.info", "Document ready: Q4 compensation letter",
     "A secure document has been sent to you for review. Please log in with your corporate email and "
     "password to view and sign the compensation letter: http://docusign-secure-view.info/sign"),
    ("accounts@vendor-supp1ies.com", "Updated banking details for invoice payment",
     "Please note that our bank details have changed. Update your records and send the outstanding "
     "payment of $18,450 to the new account below before Friday. Our phone line is down, so please do "
     "not call to confirm."),
    ("payroll@acme-corp-hr.net", "Action needed: confirm direct deposit information",
     "HR is migrating to a new payroll provider. Please sign in and confirm your bank account details "
     "by end of day so your salary is not delayed: http://hr-payroll-portal.xyz/login"),
    ("security@micros0ft-account.com", "New sign-in from an unrecognized device",
     "Someone signed in to your account from Lagos, Nigeria. If this was not you, secure your account "
     "immediately by confirming your password here: http://account-secure-login.click/verify"),
    ("tracking@dhl-parcel-delivery.top", "Your parcel is waiting: customs fee due",
     "Your parcel could not be delivered because of an unpaid customs fee of $2.99. Pay now at "
     "http://dhl-redelivery.site/pay within 24 hours or the shipment will be returned."),
    ("marcus.lee@partner-docs.online", "Re: Updated vendor contract.pdf",
     "Hi, as discussed here is the updated vendor contract. Click to open the shared file. You may be "
     "asked to sign in with your email password to access it: http://partner-docs.online/contract"),
]

EXTERNAL_SENDERS = ["accounts@partner-co.com", "orders@vendor-supplies.com", "hello@clientgroup.org",
                    "support@cloudhost.io"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fake_url(rng: random.Random, brand: str) -> str:
    """A suspicious-looking link: hyphenated brand + odd TLD, or a short link."""
    if rng.random() < 0.15:
        code = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(7))
        return f"http://bit.ly/{code}"
    slug = re.sub(r"[^a-z0-9]", "", brand.lower())
    host = f"{slug}-{rng.choice(URL_WORDS)}.{rng.choice(BAD_TLDS)}"
    return f"http://{host}/{rng.choice(URL_PATHS)}?id={rng.randint(1000, 99999)}"


def phishing_slots(rng: random.Random) -> dict:
    brand = rng.choice(BRANDS)
    return {
        "name": rng.choice(FIRST_NAMES),
        "brand": brand,
        "hours": rng.choice([2, 6, 12, 24, 24, 48, 72]),
        "amount": f"{rng.randint(2, 95) * 100:,}",
        "small": f"{rng.randint(1, 9)}.{rng.choice(['49', '95', '99'])}",
        "inv": f"INV-{rng.randint(10000, 99999)}",
        "url": fake_url(rng, brand),
        "exec": rng.choice(EXECS),
    }


def legit_slots(rng: random.Random) -> dict:
    return {
        "name": rng.choice(FIRST_NAMES),
        "sender": rng.choice(FIRST_NAMES),
        "day": rng.choice(DAYS_OF_WEEK),
        "time": rng.choice(TIMES),
        "project": rng.choice(PROJECTS),
        "dept": rng.choice(DEPTS),
        "month": rng.choice(MONTHS),
        "quarter": rng.randint(1, 4),
        "num": rng.randint(2, 12),
        "room": rng.choice(ROOMS),
        "inv": f"INV-{rng.randint(10000, 99999)}",
    }


def compose(rng: random.Random, category: dict, slots: dict) -> str:
    """Build 'subject + greeting + body + closer' as one string."""
    subject = rng.choice(category["subjects"]).format(**slots)
    greeting = rng.choice(category["greetings"]).format(**slots)
    body = rng.choice(category["bodies"]).format(**slots)
    closer = rng.choice(category.get("closers", GENERIC_LEGIT_CLOSERS)).format(**slots)
    return f"{subject} {greeting} {body} {closer}"


def make_phishing_text(rng: random.Random) -> str:
    return compose(rng, rng.choice(PHISHING_CATEGORIES), phishing_slots(rng))


def make_legit_text(rng: random.Random) -> str:
    return compose(rng, rng.choice(LEGIT_CATEGORIES), legit_slots(rng))


def generate_unique(rng: random.Random, count: int, make_fn) -> list:
    """Generate `count` texts, avoiding exact duplicates where possible."""
    seen, texts, attempts = set(), [], 0
    while len(texts) < count and attempts < count * 200:
        attempts += 1
        text = make_fn(rng)
        if text not in seen:
            seen.add(text)
            texts.append(text)
    while len(texts) < count:  # extremely unlikely fallback
        texts.append(make_fn(rng))
    return texts


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #
def build_training_data(seed: int = SEED) -> pd.DataFrame:
    """Balanced, shuffled training set: columns email_text, label."""
    rng = random.Random(seed)
    phishing = generate_unique(rng, TRAIN_PER_CLASS, make_phishing_text)
    legitimate = generate_unique(rng, TRAIN_PER_CLASS, make_legit_text)
    df = pd.DataFrame({
        "email_text": phishing + legitimate,
        "label": [1] * len(phishing) + [0] * len(legitimate),
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def write_training_csv(path=TRAIN_FILE) -> Path:
    """Create data/phishing_train.csv (also called by phishing_detector if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_training_data().to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Unlabeled demo emails
# --------------------------------------------------------------------------- #
def _employee(rng: random.Random) -> tuple:
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    return first, f"{first.lower()}.{last.lower()}@{COMPANY_DOMAIN}"


def build_sample_emails(seed: int = SEED, include_labels: bool = False) -> pd.DataFrame:
    """
    ~42 safe + 8 tricky phishing emails for the dashboard demo.

    Columns: timestamp, sender, subject, body, recipient (no label column unless
    include_labels=True, which is only meant for testing the model).
    """
    rng = random.Random(seed + 1)  # different stream from the training data
    rows = []

    # --- safe emails (some contain words like "invoice" / "password" on purpose) ---
    seen = set()
    while len(rows) < SAMPLE_SAFE_COUNT:
        subject_t, body_t = rng.choice(SAFE_SAMPLE_TEMPLATES)
        slots = legit_slots(rng)
        subject, body = subject_t.format(**slots), body_t.format(**slots)
        if (subject, body) in seen:
            continue
        seen.add((subject, body))
        _, sender = _employee(rng) if rng.random() < 0.85 else (None, rng.choice(EXTERNAL_SENDERS))
        rows.append({"sender": sender, "subject": subject, "body": body, "label": 0})

    # --- tricky phishing emails ---
    for sender, subject, body in TRICKY_PHISHING:
        rows.append({"sender": sender, "subject": subject, "body": body, "label": 1})

    # --- timestamps (3 working days), recipients, chronological order ---
    start = datetime(2026, 11, 18)
    for row in rows:
        day = start + timedelta(days=rng.randint(0, 2))
        row["timestamp"] = day + timedelta(hours=rng.randint(9, 16), minutes=rng.randint(0, 59),
                                           seconds=rng.randint(0, 59))
        row["recipient"] = _employee(rng)[1]

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime(TIMESTAMP_FORMAT)
    columns = ["timestamp", "sender", "subject", "body", "recipient"]
    return df[columns + ["label"]] if include_labels else df[columns]


def write_sample_csv(path=SAMPLE_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_sample_emails().to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    try:
        train_path = write_training_csv()
        sample_path = write_sample_csv()
    except OSError as exc:
        print(f"ERROR: could not write data files: {exc}", file=sys.stderr)
        return 1

    train = pd.read_csv(train_path)
    sample = pd.read_csv(sample_path)
    print(f"Saved {len(train)} labeled training emails to {train_path}")
    print(f"  Phishing (1): {int((train['label'] == 1).sum())}   Legitimate (0): {int((train['label'] == 0).sum())}")
    print(f"Saved {len(sample)} unlabeled demo emails to {sample_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
