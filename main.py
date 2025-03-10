from flask import Flask, jsonify, make_response, request
import mysql.connector
import bcrypt
import os
import numpy as np

import classificador as clf

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="1234",
    database="medcaredb",
    auth_plugin='mysql_native_password'
)

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:1234@localhost/medcare'

#CADASTRO DE USUARIO
@app.route('/cadastro', methods=['POST'])
def cadastrar_usuario():
    try:
        # Obtém os dados do corpo da requisição
        nome = request.json['nome']
        email = request.json['email']
        senha = request.json['senha']
        tipo_usuario = request.json['tipo_usuario']
        especialidade = request.json['especialidade']
        
        # Gera o hash da senha
        salt = bcrypt.gensalt()
        senha_hash = bcrypt.hashpw(senha.encode('utf-8'), salt)
        
        # Insere o usuário no banco de dados
        cursor = mydb.cursor()
        cursor.execute("""
            INSERT INTO Usuario (nome, email, senha, tipo_usuario, especialidade)
            VALUES (%s, %s, %s, %s, %s)
        """, (nome, email, senha_hash.decode('utf-8'), tipo_usuario, especialidade))
        
        mydb.commit()
        cursor.close()

        return jsonify({'message': 'Usuário cadastrado com sucesso!'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#LOGIN DE USUARIO

@app.route('/login', methods=['POST'])
def login_usuario():
    try:
        # Obtém os dados do corpo da requisição
        email = request.json.get('email')
        senha = request.json.get('senha')

        # Verifica se os dados foram fornecidos
        if not email or not senha:
            return jsonify({'message': 'Email e senha são obrigatórios'}), 400
        
        # Busca o usuário pelo email
        cursor = mydb.cursor()
        cursor.execute("SELECT idUsuario, nome, senha, tipo_usuario FROM Usuario WHERE email = %s", (email,))
        usuario = cursor.fetchone()

        # Fecha o cursor
        cursor.close()

        # Verifica se o usuário foi encontrado
        if not usuario:
            return jsonify({'message': 'Email ou senha incorretos'}), 401
        
        idUsuario, nome, senha_hash, tipo_usuario = usuario
        
        # Verifica a senha
        if bcrypt.checkpw(senha.encode('utf-8'), senha_hash.encode('utf-8')):
            return jsonify({'message': f'Bem-vindo(a) {nome}!', 'tipo_usuario': tipo_usuario}), 200
        else:
            return jsonify({'message': 'Email ou senha incorretos'}), 401

    except Exception as e:
        return jsonify({'error': f"Erro inesperado: {str(e)}"}), 500

@app.route('/cadastro_paciente', methods=['POST'])
def cadastrar_paciente():
    try:
        # Obtém os dados do paciente da requisição
        dados = request.get_json()

        # Validação dos dados obrigatórios
        required_fields = ['nome', 'sexo', 'data_nascimento', 'cpf', 'email', 'telefone', 'cep', 'plano']
        for field in required_fields:
            if field not in dados:
                return jsonify({'error': f'Campo {field} é obrigatório!'}), 400

        nome = dados['nome']
        sexo = dados['sexo']
        data_nascimento = dados['data_nascimento']
        cpf = dados['cpf']
        email = dados['email']
        telefone = dados['telefone']
        cep = dados['cep']
        plano = dados['plano']

        # Verifica se o email já existe
        cursor = mydb.cursor()
        cursor.execute("SELECT idPaciente FROM Paciente WHERE email = %s", (email,))
        paciente_existente = cursor.fetchone()

        if paciente_existente:
            cursor.close()
            return jsonify({'error': 'Paciente com este email já cadastrado!'}), 400

        # Insere o novo paciente no banco de dados
        cursor.execute("""
            INSERT INTO Paciente (nome, sexo, data_nascimento, cpf, email, telefone, cep, plano)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (nome, sexo, data_nascimento, cpf, email, telefone, cep, plano))

        # Confirma a inserção e pega o ID do paciente
        paciente_id = cursor.lastrowid
        mydb.commit()

        cursor.close()

        # Retorna a resposta com os dados do paciente recém-cadastrado
        response = {
            'idPaciente': paciente_id,
            'nome': nome,
            'sexo': sexo,
            'data_nascimento': data_nascimento,
            'cpf': cpf,
            'email': email,
            'telefone': telefone,
            'cep': cep,
            'plano': plano
        }

        return jsonify(response), 201

    except Exception as e:
        # Se ocorrer algum erro, retorna a mensagem de erro
        return jsonify({'error': str(e)}), 500


#PUXA TODOS OS PACIENTES
@app.route('/pacientes', methods=['GET'])
def get_pacientes():
    try:
        # Obtém o cursor e executa a consulta
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM paciente")
        
        # Lê todos os resultados da consulta
        pacientes = cursor.fetchall()
        
        # Retorna os resultados como resposta (por exemplo, em formato JSON)
        return jsonify(pacientes)
    except mysql.connector.errors.InternalError as e:
        # Retorna uma mensagem de erro se ocorrer um problema
        return jsonify({'error': str(e)}), 500
    finally:
        # Garante que o cursor e a conexão sejam fechados corretamente
        cursor.close()
        mydb.close()


## ROTA DA IA

@app.route('/clf', methods=['POST'])
def classificar_imagem():
    try:
        if 'imagem' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada!'}), 400

        file = request.files['imagem']

        # Obtém o email do paciente da request
        email_paciente = request.form.get('email')
        if not email_paciente:
            return jsonify({'error': 'Email do paciente não fornecido!'}), 400

        # Conexão com o banco de dados
        cursor = mydb.cursor()

        # Verifica se o paciente existe no banco de dados pelo email
        cursor.execute("SELECT idPaciente FROM Paciente WHERE email = %s", (email_paciente,))
        paciente = cursor.fetchone()

        if not paciente:
            return jsonify({'error': 'Paciente não encontrado!'}), 404

        paciente_id = paciente[0]  # Obtém o ID do paciente encontrado

        temp_dir = './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Salva a imagem temporariamente
        img_path = os.path.join(temp_dir, file.filename)
        file.save(img_path)

        # Preprocessa a imagem
        img = clf.preprocess_image_with_generator(img_path, target_size=(320, 320))

        # Faz a previsão com o modelo
        predictions = clf.model.predict(img)

        # Cria um diagnóstico provisório (não revisado)
        resultado = clf.labels[np.argmax(predictions[0])]  # Resultado com base na classe com maior probabilidade
        probabilidade = float(np.max(predictions[0]))  # Converter para float padrão

        # 1. Cria o Diagnóstico no banco de dados
        cursor.execute("""
            INSERT INTO Diagnostico (resultado, probabilidade)
            VALUES (%s, %s)
        """, (resultado, probabilidade))
        diagnostico_id = cursor.lastrowid  # Obtém o ID do diagnóstico inserido

        # 2. Cria o Exame no banco de dados, associando o Diagnóstico e o Paciente encontrado
        cursor.execute("""
            INSERT INTO Exame (data_exame, imagem_exame, Diagnostico_idDiagnostico, Usuario_idUsuario, Paciente_idPaciente)
            VALUES (NOW(), %s, %s, %s, %s)
        """, ('file-exame', diagnostico_id, 13, paciente_id))  # Usuário com ID 13, altere conforme necessário
        exame_id = cursor.lastrowid  # Obtém o ID do exame inserido

        # Confirma a transação (com o exame e o diagnóstico)
        mydb.commit()

        # Fecha o cursor
        cursor.close()

        # Resposta com os resultados da classificação
        response = {
            'diagnostico': resultado,
            'probabilidade': f"{probabilidade * 100:.2f}%",
            'detalhes': {label: f"{float(pred * 100):.2f}%" for label, pred in zip(clf.labels, predictions[0])}
        }

        # Remove a imagem temporária após o processamento
        os.remove(img_path)

        return jsonify(response), 200

    except Exception as e:
        # Certifica-se de fechar o cursor e retornar erro
        if 'cursor' in locals() and cursor:
            cursor.close()
        return jsonify({'error': str(e)}), 500


#PUXA TODOS OS PACIENTES
@app.route('/exames', methods=['GET'])
def get_exames():
    try:
        # Obtém o cursor e executa a consulta
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM exame")
        
        # Lê todos os resultados da consulta
        exames = cursor.fetchall()
        
        # Retorna os resultados como resposta (por exemplo, em formato JSON)
        return jsonify(exames)
    except mysql.connector.errors.InternalError as e:
        # Retorna uma mensagem de erro se ocorrer um problema
        return jsonify({'error': str(e)}), 500
    finally:
        # Garante que o cursor e a conexão sejam fechados corretamente
        cursor.close()
        mydb.close()


app.run(debug=True)