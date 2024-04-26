'''
 # @ Author: Mitas Ray
 # @ Create date: 2024-01-29
 # @ Modified by: Mitas Ray
 # @ Modified date: 2024-03-29
 # @ Description: Used to send an email to a list of recipients with the training log attached. This allows the 
 the recipients to view the training log without having to ssh into the VM where the training occurs.
 '''
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from automated_training_auxiliary_variables import EMAIL_RECIPIENTS_FOR_LOGS
from automated_training_auxiliary_functions import send_email, check_that_model_is_supported


def send_training_log(attachment_path, recipients: list, model: str, message: str):
    check_that_model_is_supported(model)

    def get_filename_from_path(path):
        '''Find the last occurrence of a slash and then keep the rest of the string.'''
        last_slash_index = path.rfind('/')    # rfind starts searching from the right side (hence the 'r' in `rfind`) and returns the index of the last occurrence of the specified substring
        assert last_slash_index != -1, f'No "/" found in {path}'    # if '/' is not found, then `rfind(...)` returns -1
        return path[last_slash_index + 1:]

    print(f'Sending email to {recipients}')
    sender_email = 'notifications@ficc.ai'
    
    msg = MIMEMultipart()
    msg['Subject'] = f'Training log for {model} model trained today'
    msg['From'] = sender_email

    body = MIMEText(message, 'plain')
    msg.attach(body)

    attachment_filename = get_filename_from_path(attachment_path)
    with open(attachment_path, 'rb') as attachment:    # attach the file
        part = MIMEApplication(attachment.read())
        part.add_header('Content-Disposition', 'attachment', filename=attachment_filename)
        msg.attach(part)
    send_email(sender_email, msg, recipients)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: $ python send_email_with_training_log.py <filepath> <model> <message>')
    else:
        filepath = sys.argv[1]
        model = sys.argv[2]
        message = sys.argv[3]
        print(f'Sending email with {filepath}')
        send_training_log(filepath, EMAIL_RECIPIENTS_FOR_LOGS, model, message)
