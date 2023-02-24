import getpass

import psycopg2
import sshtunnel

ssh_host = input('SSH Host: ')
ssh_user = input('SSH Username: ')
ssh_pass = getpass.getpass('SSH Password: ')
db_host = input('DB Host: ')

with sshtunnel.open_tunnel((ssh_host, 22), ssh_username=ssh_user, ssh_password=ssh_pass,
                           remote_bind_address=(db_host, 5432)) as tunnel:

    db_user = input('PGSql Username: ')
    db_pass = getpass.getpass('PGSql Password: ')
    conn = psycopg2.connect(database='capstone', user=db_user, password=db_pass,
                            port=tunnel.local_bind_port)

    query = 'SELECT 1'
    cursor = conn.cursor()
    cursor.execute(query)

    print(cursor.fetchall())
