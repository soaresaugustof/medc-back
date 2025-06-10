import mysql.connector
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()

# Lê o script SQL
with open(os.path.join('db', 'db.sql'), 'r', encoding='utf-8') as f:
    sql_script = f.read()

# Conecta ao banco de dados MySQL
mysql_port = os.getenv("MYSQLPORT") or 3306
conn = mysql.connector.connect(
    host=os.getenv("MYSQLHOST"),
    user=os.getenv("MYSQLUSER"),
    passwd=os.getenv("MYSQL_ROOT_PASSWORD"),
    port=int(mysql_port),
    # Não seleciona database para permitir CREATE SCHEMA
)
cursor = conn.cursor()

# Executa o script SQL (pode conter múltiplos comandos)
for result in cursor.execute(sql_script, multi=True):
    pass  # Apenas executa cada comando

cursor.close()
conn.close()
print("Script de criação das tabelas executado com sucesso!")
