import mysql.connector

def connect_db():
    """ Connect to closure database."""
    
    mydb = mysql.connector.connect(
      host="XXX",
      user="XXX",
      password="XXX",
      database="XXX"
    )
    
    return(mydb)