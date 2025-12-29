from email.base64mime import body_decode

import pandas as pd
import re
import nltk
from aiohttp.web_urldispatcher import Domain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def word_cloud(data):
    #Pre_process
    phishing_subject_text = " ".join(data[data["label"] == 1]["subject"].str.lower())
    phishing_body_text = " ".join(data[data["label"] == 1]["body"].str.lower())

    #Lowercase
    phishing_subject_text = phishing_subject_text.lower()
    phishing_body_text = phishing_body_text.lower()

    #Remove punctuation and numbers
    phishing_subject_text = re.sub(r'[^a-z\s]', '', phishing_subject_text)
    phishing_body_text = re.sub(r'[^a-z\s]', '', phishing_body_text)

    #Tokenise
    phishing_subject_tokens = phishing_subject_text.split()
    phishing_body_tokens = phishing_body_text.split()

    phishing_subject_tokens = [lemmatizer.lemmatize(word) for word in phishing_subject_tokens if word not in stop_words]
    phishing_body_tokens = [lemmatizer.lemmatize(word) for word in phishing_body_tokens if word not in stop_words]

    phishing_subject_tokens = " ".join(phishing_subject_tokens)
    phishing_body_tokens = " ".join(phishing_body_tokens)

    #Generate word clouds
    phishing_subject_wc = WordCloud(width=800, height=400, background_color="white").generate(phishing_subject_tokens)
    phishing_body_wc = WordCloud(width=800, height=400, background_color="white").generate(phishing_body_tokens)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(phishing_subject_wc, interpolation="bilinear")
    axes[0].set_title("Phishing Subjects")
    axes[0].axis("off")

    axes[1].imshow(phishing_body_wc, interpolation="bilinear")
    axes[1].set_title("Phishing Body")
    axes[1].axis("off")

    plt.show()


def percentage_capitalisation(data):
    phishing_subject_capitalisation_percentage_total = 0
    legitimate_subject_capitalisation_percentage_total = 0
    phishing_body_capitalisation_percentage_total = 0
    legitimate_body_capitalisation_percentage_total = 0
    phishing_subject_word_capitalisation_percentage_total = 0
    legitimate_subject_word_capitalisation_percentage_total = 0
    phishing_body_word_capitalisation_percentage_total = 0
    legitimate_body_word_capitalisation_percentage_total = 0
    phish_counter = 0
    legit_counter = 0
    for row in data.iterrows():
        row = row[1]
        subject_capital_count = sum(1 for char in row["subject"] if char.isupper())
        subject_letter_count = sum(1 for char in row["subject"] if char.isalpha())
        subject_capitalisation_percentage = 0

        subject_words = row["subject"].split()
        subject_word_capital_count = sum(1 for word in subject_words if word.isupper())
        subject_word_count = len(subject_words)
        subject_word_capitalisation_percentage = 0

        body_capital_count = sum(1 for char in row["body"] if char.isupper())
        body_letter_count = sum(1 for char in row["body"] if char.isalpha())
        body_capitalisation_percentage = 0

        body_words = row["body"].split()
        body_word_capital_count = sum(1 for word in body_words if word.isupper())
        body_word_count = len(body_words)
        body_word_capitalisation_percentage = 0

        if subject_letter_count == 0 or body_letter_count == 0 or row["body_missing"] == 1 or row["subject_missing"] == 1:
            continue
        else:
            subject_capitalisation_percentage = subject_capital_count / subject_letter_count
            body_capitalisation_percentage = body_capital_count / body_letter_count
            subject_word_capitalisation_percentage = subject_word_capital_count / subject_word_count
            body_word_capitalisation_percentage = body_word_capital_count / body_word_count

            if row["label"] == 1:
                phishing_subject_capitalisation_percentage_total += subject_capitalisation_percentage
                phishing_body_capitalisation_percentage_total += body_capitalisation_percentage
                phishing_subject_word_capitalisation_percentage_total += subject_word_capitalisation_percentage
                phishing_body_word_capitalisation_percentage_total += body_word_capitalisation_percentage
                phish_counter += 1
            else:
                legitimate_subject_capitalisation_percentage_total += subject_capitalisation_percentage
                legitimate_body_capitalisation_percentage_total += body_capitalisation_percentage
                legitimate_subject_word_capitalisation_percentage_total += subject_word_capitalisation_percentage
                legitimate_body_word_capitalisation_percentage_total += body_word_capitalisation_percentage
                legit_counter += 1
    phishing_subject_capitalisation_percentage_avg = (phishing_subject_capitalisation_percentage_total / phish_counter) * 100
    legitimate_subject_capitalisation_percentage_avg = (legitimate_subject_capitalisation_percentage_total / legit_counter) * 100
    phishing_body_capitalisation_percentage_avg = (phishing_body_capitalisation_percentage_total / phish_counter) * 100
    legitimate_body_capitalisation_percentage_avg = (legitimate_body_capitalisation_percentage_total / legit_counter) * 100
    phishing_subject_word_capitalisation_percentage_avg = (phishing_subject_word_capitalisation_percentage_total / phish_counter) * 100
    legitimate_subject_word_capitalisation_percentage_avg = (legitimate_subject_word_capitalisation_percentage_total / legit_counter) * 100
    phishing_body_word_capitalisation_percentage_avg = (phishing_body_word_capitalisation_percentage_total / phish_counter) * 100
    legitimate_body_word_capitalisation_percentage_avg = (legitimate_body_word_capitalisation_percentage_total / legit_counter) * 100
    print(f"------- Average capitalisation percentage of phishing emails: -------")
    print(f"Phishing Subject Letter Avg:   {phishing_subject_capitalisation_percentage_avg}%")
    print(f"Legitimate Subject Letter Avg: {legitimate_subject_capitalisation_percentage_avg}%")
    print(f"Phishing Body Letter Avg:      {phishing_body_capitalisation_percentage_avg}%")
    print(f"Legitimate Body Letter Avg:    {legitimate_body_capitalisation_percentage_avg}%")
    print(f"Phishing Subject Word Avg:   {phishing_subject_word_capitalisation_percentage_avg}%")
    print(f"Legitimate Subject Word Avg: {legitimate_subject_word_capitalisation_percentage_avg}%")
    print(f"Phishing Body Word Avg:      {phishing_body_word_capitalisation_percentage_avg}%")
    print(f"Legitimate Body Word Avg:    {legitimate_body_word_capitalisation_percentage_avg}%")

def probability_of_null_body_or_subject(data):
    total_null_body = 0
    body_total_phishing_emails = 0
    body_total_legitimate_emails = 0
    total_null_subj = 0
    subj_total_phishing_emails = 0
    subj_total_legitimate_emails = 0
    for row in data.iterrows():
        row = row[1]
        if row["body_missing"] == 1:
            total_null_body += 1
            print(f"Null body: Subj: {row['subject']}")
            if row["label"] == 1:
                body_total_phishing_emails += 1
            else:
                body_total_legitimate_emails += 1
        if row["subject_missing"] == 1:
            total_null_subj += 1
            if row["label"] == 1:
                subj_total_phishing_emails += 1
            else:
                subj_total_legitimate_emails += 1
    phishing_email_percent = (body_total_phishing_emails + subj_total_phishing_emails) / (total_null_body + total_null_subj) * 100
    body_phishing_email_percent = body_total_phishing_emails / total_null_body * 100
    subj_phishing_email_percent = subj_total_phishing_emails / total_null_subj * 100
    print("\n")
    print(f"------- Probability of null body or subject correlation to phishing emails: -------")
    print(f"Null Phishing emails: {phishing_email_percent}%")
    print(f"Null Legitimate emails: {100 - phishing_email_percent}%")

    print(f"Null body phishing emails: {body_phishing_email_percent}%")
    print(f"Null body legitimate emails: {100 - body_phishing_email_percent}%")

    print(f"Null subject phishing emails: {subj_phishing_email_percent}%")
    print(f"Null subject legitimate emails: {100 - subj_phishing_email_percent}%")

def avg_length_of_body(data):
    phishing_emails = 0
    phishing_emails_length = 0
    phishing_emails_subject_length = 0
    legitimate_emails = 0
    legitimate_emails_length = 0
    legitimate_emails_subject_length = 0
    for row in data.iterrows():
        row = row[1]
        if row["label"] == 1:
            phishing_emails += 1
            phishing_emails_length += len(row["body"])
            phishing_emails_subject_length += len(row["subject"])
        else:
            legitimate_emails += 1
            legitimate_emails_length += len(row["body"])
            legitimate_emails_subject_length += len(row["subject"])
    print("\n")
    print(f"------- Average length of phishing emails: -------")
    print(f"Average length of phishing emails: {phishing_emails_length / phishing_emails}")
    print(f"Average length of legitimate emails: {legitimate_emails_length / legitimate_emails}")
    print(f"Average length of phishing emails subject: {phishing_emails_subject_length / phishing_emails}")
    print(f"Average length of legitimate emails subject: {legitimate_emails_subject_length / legitimate_emails}")

def use_of_characters(data):
    class Char_stats:
        def __init__(self):
            self.phishing_body = 0
            self.phishing_subj = 0
            self.phishing_email = 0
            self.total_body = 0
            self.total_subj = 0
            self.total = 0
    exclamation_mark = Char_stats()
    urgent = Char_stats()
    money = Char_stats()
    please = Char_stats()
    vital = Char_stats()
    must = Char_stats()
    account = Char_stats()
    you = Char_stats()
    sir = Char_stats()
    madame = Char_stats()
    free = Char_stats()
    suspicious = Char_stats()
    for row in data.iterrows():
        row = row[1]
        body_subject_lower = str(row["subject"]).lower() + str(row["body"]).lower()
        body_lower = str(row["body"]).lower()
        subj_lower = str(row["subject"]).lower()

        #Exclamation mark
        if "!" in body_subject_lower:
            exclamation_mark.total += 1
            if "!" in body_lower:
                exclamation_mark.total_body += 1
            if "!" in subj_lower:
                exclamation_mark.total_subj += 1
            if row["label"] == 1:
                exclamation_mark.phishing_email += 1
                if "!" in body_lower:
                    exclamation_mark.phishing_body += 1
                if "!" in subj_lower:
                    exclamation_mark.phishing_subj += 1

        # urgent
        if "urgent" in body_subject_lower:
            urgent.total += 1
            if "urgent" in body_lower:
                urgent.total_body += 1
            if "urgent" in subj_lower:
                urgent.total_subj += 1
            if row["label"] == 1:
                urgent.phishing_email += 1
                if "urgent" in body_lower:
                    urgent.phishing_body += 1
                if "urgent" in subj_lower:
                    urgent.phishing_subj += 1

        # money
        if "money" in body_subject_lower:
            money.total += 1
            if "money" in body_lower:
                money.total_body += 1
            if "money" in subj_lower:
                money.total_subj += 1
            if row["label"] == 1:
                money.phishing_email += 1
                if "money" in body_lower:
                    money.phishing_body += 1
                if "money" in subj_lower:
                    money.phishing_subj += 1

        # please
        if "please" in body_subject_lower:
            please.total += 1
            if "please" in body_lower:
                please.total_body += 1
            if "please" in subj_lower:
                please.total_subj += 1
            if row["label"] == 1:
                please.phishing_email += 1
                if "please" in body_lower:
                    please.phishing_body += 1
                if "please" in subj_lower:
                    please.phishing_subj += 1

        # vital
        if "vital" in body_subject_lower:
            vital.total += 1
            if "vital" in body_lower:
                vital.total_body += 1
            if "vital" in subj_lower:
                vital.total_subj += 1
            if row["label"] == 1:
                vital.phishing_email += 1
                if "vital" in body_lower:
                    vital.phishing_body += 1
                if "vital" in subj_lower:
                    vital.phishing_subj += 1

        # must
        if "must" in body_subject_lower:
            must.total += 1
            if "must" in body_lower:
                must.total_body += 1
            if "must" in subj_lower:
                must.total_subj += 1
            if row["label"] == 1:
                must.phishing_email += 1
                if "must" in body_lower:
                    must.phishing_body += 1
                if "must" in subj_lower:
                    must.phishing_subj += 1

        # account
        if "account" in body_subject_lower:
            account.total += 1
            if "account" in body_lower:
                account.total_body += 1
            if "account" in subj_lower:
                account.total_subj += 1
            if row["label"] == 1:
                account.phishing_email += 1
                if "account" in body_lower:
                    account.phishing_body += 1
                if "account" in subj_lower:
                    account.phishing_subj += 1

        # you
        if "you" in body_subject_lower:
            you.total += 1
            if "you" in body_lower:
                you.total_body += 1
            if "you" in subj_lower:
                you.total_subj += 1
            if row["label"] == 1:
                you.phishing_email += 1
                if "you" in body_lower:
                    you.phishing_body += 1
                if "you" in subj_lower:
                    you.phishing_subj += 1

        # sir
        if "sir" in body_subject_lower:
            sir.total += 1
            if "sir" in body_lower:
                sir.total_body += 1
            if "sir" in subj_lower:
                sir.total_subj += 1
            if row["label"] == 1:
                sir.phishing_email += 1
                if "sir" in body_lower:
                    sir.phishing_body += 1
                if "sir" in subj_lower:
                    sir.phishing_subj += 1

        # madame
        if "madame" in body_subject_lower:
            madame.total += 1
            if "madame" in body_lower:
                madame.total_body += 1
            if "madame" in subj_lower:
                madame.total_subj += 1
            if row["label"] == 1:
                madame.phishing_email += 1
                if "madame" in body_lower:
                    madame.phishing_body += 1
                if "madame" in subj_lower:
                    madame.phishing_subj += 1

        # free
        if "free" in body_subject_lower:
            free.total += 1
            if "free" in body_lower:
                free.total_body += 1
            if "free" in subj_lower:
                free.total_subj += 1
            if row["label"] == 1:
                free.phishing_email += 1
                if "free" in body_lower:
                    free.phishing_body += 1
                if "free" in subj_lower:
                    free.phishing_subj += 1

        # suspicious
        if "suspicious" in body_subject_lower:
            suspicious.total += 1
            if "suspicious" in body_lower:
                suspicious.total_body += 1
            if "suspicious" in subj_lower:
                suspicious.total_subj += 1
            if row["label"] == 1:
                suspicious.phishing_email += 1
                if "suspicious" in body_lower:
                    suspicious.phishing_body += 1
                if "suspicious" in subj_lower:
                    suspicious.phishing_subj += 1

    print("\n")
    print(f"------- Use of characters correlation to phishing emails: -------")
    print(f"Phishing percent of emails with exclamation mark: {exclamation_mark.phishing_email / exclamation_mark.total * 100}%")
    print(f"Phishing percent of body with exclamation mark: {exclamation_mark.phishing_body / exclamation_mark.total_body * 100}%")
    print(f"Phishing percent of subject with exclamation mark: {exclamation_mark.phishing_subj / exclamation_mark.total_subj * 100}%")
    print(f"Total: {exclamation_mark.total} Body: {exclamation_mark.total_body} Subject: {exclamation_mark.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with urgent: {urgent.phishing_email / urgent.total * 100}%")
    print(f"Phishing percent of body with urgent: {urgent.phishing_body / urgent.total_body * 100}%")
    print(f"Phishing percent of subject with urgent: {urgent.phishing_subj / urgent.total_subj * 100}%")
    print(f"Total: {urgent.total} Body: {urgent.total_body} Subject: {urgent.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with money: {money.phishing_email / money.total * 100}%")
    print(f"Phishing percent of body with money: {money.phishing_body / money.total_body * 100}%")
    print(f"Phishing percent of subject with money: {money.phishing_subj / money.total_subj * 100}%")
    print(f"Total: {money.total} Body: {money.total_body} Subject: {money.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with please: {please.phishing_email / please.total * 100}%")
    print(f"Phishing percent of body with please: {please.phishing_body / please.total_body * 100}%")
    print(f"Phishing percent of subject with please: {please.phishing_subj / please.total_subj * 100}%")
    print(f"Total: {please.total} Body: {please.total_body} Subject: {please.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with vital: {vital.phishing_email / vital.total * 100}%")
    print(f"Phishing percent of body with vital: {vital.phishing_body / vital.total_body * 100}%")
    print(f"Phishing percent of subject with vital: {vital.phishing_subj / vital.total_subj * 100}%")
    print(f"Total: {vital.total} Body: {vital.total_body} Subject: {vital.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with must: {must.phishing_email / must.total * 100}%")
    print(f"Phishing percent of body with must: {must.phishing_body / must.total_body * 100}%")
    print(f"Phishing percent of subject with must: {must.phishing_subj / must.total_subj * 100}%")
    print(f"Total: {must.total} Body: {must.total_body} Subject: {must.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with account: {account.phishing_email / account.total * 100}%")
    print(f"Phishing percent of body with account: {account.phishing_body / account.total_body * 100}%")
    print(f"Phishing percent of subject with account: {account.phishing_subj / account.total_subj * 100}%")
    print(f"Total: {account.total} Body: {account.total_body} Subject: {account.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with you: {you.phishing_email / you.total * 100}%")
    print(f"Phishing percent of body with you: {you.phishing_body / you.total_body * 100}%")
    print(f"Phishing percent of subject with you: {you.phishing_subj / you.total_subj * 100}%")
    print(f"Total: {you.total} Body: {you.total_body} Subject: {you.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with sir: {sir.phishing_email / sir.total * 100}%")
    print(f"Phishing percent of body with sir: {sir.phishing_body / sir.total_body * 100}%")
    print(f"Phishing percent of subject with sir: {sir.phishing_subj / sir.total_subj * 100}%")
    print(f"Total: {sir.total} Body: {sir.total_body} Subject: {sir.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with madame: {madame.phishing_email / madame.total * 100}%")
    print(f"Phishing percent of body with madame: {madame.phishing_body / madame.total_body * 100}%")
    print(f"Phishing percent of subject with madame: {madame.phishing_subj / madame.total_subj * 100}%")
    print(f"Total: {madame.total} Body: {madame.total_body} Subject: {madame.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with free: {free.phishing_email / free.total * 100}%")
    print(f"Phishing percent of body with free: {free.phishing_body / free.total_body * 100}%")
    print(f"Phishing percent of subject with free: {free.phishing_subj / free.total_subj * 100}%")
    print(f"Total: {free.total} Body: {free.total_body} Subject: {free.total_subj}")
    print("\n")

    print(f"Phishing percent of emails with suspicious: {suspicious.phishing_email / suspicious.total * 100}%")
    print(f"Phishing percent of body with suspicious: {suspicious.phishing_body / suspicious.total_body * 100}%")
    print(f"Phishing percent of subject with suspicious: {suspicious.phishing_subj / suspicious.total_subj * 100}%")
    print(f"Total: {suspicious.total} Body: {suspicious.total_body} Subject: {suspicious.total_subj}")
    print("\n")

def links_correlations(data):
    phishing_emails = 0
    phishing_emails_links = 0
    for row in data.iterrows():
        row = row[1]
        if row["label"] == 1:
            phishing_emails += 1
            #urls is weird some entries are 1 some are the actual urls. Also its not accurate
            if "http" in str(row["urls"]).lower():
                phishing_emails_links += 1
    print(f"Percentage of phishing emails with links: {phishing_emails_links / phishing_emails * 100}%")

def domain_correlations(data):
    class Domain:
        def __init__(self, domain):
            self.domain = domain
            self.count = 0
            self.phishing_count = 0

    domain_suffix_list = []

    for row in data.iterrows():
        row = row[1]
        sender = str(row["sender"])
        email_address = sender.split("<")[-1].split(">")[0]
        domain_suffix = email_address.split(".")[-1]
        # check if this domain already exists
        existing = None
        for d in domain_suffix_list:
            if d.domain == domain_suffix:
                existing = d
                break

        if existing is None:
            d = Domain(domain_suffix)
            d.count = 1
            if row["label"] == 1:
                d.phishing_count = 1
            domain_suffix_list.append(d)
        else:
            existing.count += 1
            if row["label"] == 1:
                existing.phishing_count += 1

    for d in domain_suffix_list:
        print(f"{d.domain}: {d.count} phishing emails: {d.phishing_count} ({d.phishing_count / d.count * 100}%)")




if __name__ == "__main__":
    data = pd.read_csv("Data/ProcessedData/all_data.csv")
    seven_column_data = pd.read_csv("Data/ProcessedData/seven_column_data.csv")
    percentage_capitalisation(data)
    #word_cloud(data)
    probability_of_null_body_or_subject(data)
    avg_length_of_body(data)
    use_of_characters(data)

    #7 column data
    #Its weirdly formatted and inconsistent. Might not actually use it
    #links_correlations(seven_column_data)
    #domain_correlations(seven_column_data)

    #Maybe add spell check?
