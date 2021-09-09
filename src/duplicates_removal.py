from textpack import tp

def duplicates_per_org(emails_orgs):
    """ Run the TextPack algorithm to group similar emails with a threshold of 0.95
    per organization.
    Add a column called "duplicate" to the dataframe.
    
    Input:
        emails_orgs: DataFrame
            contains as columns at least:
                - organization_id: integer
                - only_body: str
    """
    
    org_ids = list(emails_orgs.organization_id.unique())
    dic = {}
    for id_ in org_ids:
        subset = emails_orgs[emails_orgs["organization_id"] == id_]
        txtp = tp.TextPack(subset, "only_body", match_threshold=0.95, ngram_remove=r'[,-/;]')
        try:
            txtp.build_group_lookup()
            dic = {**dic, **txtp.group_lookup}
        except:
            pass

    emails_orgs['duplicate'] = emails_orgs["only_body"].map(dic).fillna(emails_orgs['only_body'])
    
    return(emails_orgs)

def is_different(only_body, duplicate):
    """ Return 0 if the content (only_body) is the same as the TextPack duplicate, 
    1 otherwise.
    
    Input:
        - only_body: str
        - duplicate: str
            "duplicate" column from duplicates_per_org()
    """
    
    if only_body == duplicate:
        return(0)
    else: 
        return(1)