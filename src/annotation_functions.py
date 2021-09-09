import re

def get_label(row_status, row_action):
    """ Match the customer sercive actions with the corresponding label.
    Label 0: Solved
    Label 1: Not found
    Label 2: More information required
    Label 3: We don't know
    
    Inputs:
        - row_status: integer
            status ID of the email ticket
        - row_action: integer
            action ID of the email ticket
    """
    
    actions_more_info = [1,6,16,17,19]
    actions_solved = [2,3,4,12,14,18]
    
    #Not found
    if row_status == 7:
        return(1)
    #More information required
    elif row_action in actions_more_info:
        return(2)
    #Solved
    elif row_action in actions_solved:
        return(0)
    else:
        return(3)
    
def fix_label_charity(row_org_id, row_label, row_body, org_id_char_jour):
    """ Swith label 1 (not found) to label 2 (more information required)
    if the organiation is a charity or journal and the email contains
    a "?" in its body. This is because charities often ask for more information
    but the customer support makes the same actions as if there was no question
    asked.
    
    Inputs:
        - row_org_id: integer
            organization ID
        - row_label: integer 
            label (0, 1, 2, 3)
        - row_body: str
            email content
        - org_id_char_jour: list
            organizations IDs that are either a charity or a journal
    """
    
    if row_org_id in org_id_char_jour and row_label == 1 and '?' in row_body:
        return(2)
    else:
        return(row_label)
    
def find_auto_emails(row):
    """ Return 1 if the email contains at least one pattern, 0 otherwise.
    
    Input:
        - row: str
            email body
    """
    
    patterns = ["we doen ons best altijd binnen drie werkdagen je vraag te beantwoorden", 
                "we streven ernaar je vraag uiterlijk binnen 3 werkdagen te beantwoorden"]
    count = 0
    for pattern in patterns:
        if bool(re.search(pattern, row.lower().replace("\n", " "))):
            count += 1
        else:
            pass
    if count > 0:
        return(1)
    else:
        return(0)
    
def group_emails(emails_orgs):
    """ Run the TextPack algorithm to group similar emails with a threshold of 0.70.
    Add a column called "groups" to the dataframe.
    
    Input:
        emails_orgs: DataFrame
            contains as column at least:
                - only_body: str
    """
    
    dic = {}
    txtp = tp.TextPack(emails_orgs, "only_body", match_threshold=0.7, ngram_remove=r'[,-/;]')
    try:
        txtp.build_group_lookup()
        dic = {**dic, **txtp.group_lookup}
    except:
        pass

    emails_orgs['groups'] = emails_orgs["only_body"].map(dic).fillna(emails_orgs['only_body'])
    return(emails_orgs)

def fix_seagulls(row):
    """ Replace "seagulls" by "Meeuwsen".
    
    Input: 
        - row: str
            email body in translated-English from Dutch 
    """
    
    if "seagulls" in row.lower():
        row = row.replace("seagulls", "Meeuwsen")
        
    return(row)

def fix_mouettes(row):
    """ Replace "mouettes" by "Meeuwsen".
    
    Input: 
        - row: str
            email body in translated-French from Dutch 
    """
    
    if "mouettes" in row.lower():
        row = row.replace("mouettes", "Meeuwsen")
        
    return(row)