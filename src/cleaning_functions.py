import re
from langdetect import detect


def remove_history(row):
    """ Remove history of previous emails that start with a certain pattern.
    
    Input:
        - row: str
    """
    
    substrings = ["van:", "from:", "*van*", "> op", "(*de {0,3}:)", 
                  "(op.{0,100}<.{0,40}@.+>)", 
                  "(\S*mail.{0,100}\[.{0,100}@closure-services.nl.*\])"]
    
    #Regex6 format:
    #Op ma 21 sep. 2020 11:26 schreef Closure <[dossiernr-www@closure-
    #services.nl](mailto:dossiernr-www@closure-services.nl)>:  
    #Regex7 format:
    #**E-mail** : [dossiernr-www@closure-services.nl]
    
    stripped = row
    
    for substring in substrings:
        stripped = re.split(re.escape(substring), stripped, flags=re.IGNORECASE)[0]
    
    return(stripped)

def remove_after_regards(row):
    """ Remove what follows after a certain patern (best regards) including the pattern.
    "Met" is not included in the patterns because sometimes it does not
    appear. Hence, if there is a "met" in the last 20 characters after removing
    the pattern, the "met" is also removed.
    
    Input:
        - row: str
    """
    
    #Start with the most common
    substring1 = "vriendelijke groet"
    substrings = ["hartelijke groet", "hartelijke en warme groeten", "groeten", "groet", "mvg"]

    stripped = re.split(substring1, row, flags=re.IGNORECASE)[0]
    for substring in substrings:
        stripped = re.split(substring, stripped, flags=re.IGNORECASE)[0]
    
    #Remove the "met" at the end of the string
    if "met" in stripped[-20:].lower():
        stripped = re.split("met", stripped, flags=re.IGNORECASE)[0:-1]
        stripped = ' '.join(stripped)
        
    return(stripped)

def remove_url_emails(row):
    """ Remove URL and email adresses.
    
    Input:
        - row: str
    """
    
    #url
    pattern = '\S*://\S*\s?'
    row = re.sub(pattern, "", row)
    # emails
    pattern = '\S*@\S*\s?'
    row = re.sub(pattern, "", row)
    
    return(row)

def remove_before_dear(row):
    """ Remove everything before a pattern (dear) but not the pattern itself.
    The pattern has to be in the first 200 characters.
    
    Input:
        - row: str
    """
    
    #Start with the most common
    substrings = ["([Bb]este [A-Z;a-z]+ ?,* ?|[Bb]este, ?|[Bb]este ?\r*)", 
                  "([Gg]eachte, ?|[Gg]eachte ?\r*|[Gg]eachte [A-Z;a-z]+ ?,* ?)", "(mevrouw,)", "(meeuwissen,)", 
                  "(hallo ?\r+)"]
    stripped = row

    for substring in substrings:
        stripped = re.split(substring, stripped, flags=re.IGNORECASE)
        if '' in stripped:
            stripped.remove('')
        
        #There is a header to remove and the pattern has been found at the begining of the email (200 first characteres)
        if len(stripped) > 1 and bool(re.search(substring, stripped[0])) == False and len(stripped[0]) < 200: 
            stripped = stripped[1:]
            stripped = "".join(stripped)
            
        else:
            stripped = "".join(stripped)
            
    return(stripped)

def remove_space_end(row):
    """ Remove space at the end of row if there is one.
    
    Input:
        - row: str
    """
    
    return(row.rstrip())

def check_Dutch(row):
    """ Return 1 is the language is at 100% Dutch and 0 otherwise.
    
    Input:
        - row: str
    """
    
    try:
        language = detect(row)
        if language != 'nl':
            return(0)
        else:
            return(1)
    except:
        return(0)            
    
    