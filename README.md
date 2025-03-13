<<<<<<< HEAD
# Aplicativos-IA
=======
# Central de Aplicativos - Rede Neural Ensinar

Este é um projeto Django que funciona como uma central de aplicativos, onde cada app possui funcionalidades específicas. O primeiro app, **redeneuralensinar**, tem como objetivo oferecer conteúdos e ferramentas para o ensino de redes neurais e matemática.

## Sumário

- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração e Execução](#configuração-e-execução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Contribuições](#contribuições)
- [Licença](#licença)
- [Contato](#contato)

## Tecnologias Utilizadas

- Python 3.x
- Django 4.x (ou a versão compatível instalada)
- Ambiente virtual (venv)

## Requisitos

- Python 3 instalado
- pip instalado

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone https://seu-repositorio.git
   cd centraldeapps

# Criação do ambiente virtual
python -m venv env

# Ativação do ambiente
# No Linux/macOS:
source env/bin/activate

# No Windows (PowerShell):
.\env\Scripts\activate

pip install -r requirements.txt

pip install django

python manage.py makemigrations
python manage.py migrate

python manage.py createsuperuser

python manage.py runserver

Acesse a aplicação:

App Rede Neural Ensinar:
http://127.0.0.1:8000/redeneuralensinar/

Área Administrativa do Django:
http://127.0.0.1:8000/admin/

centraldeapps/
├── manage.py
├── centraldeapps/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
└── redeneuralensinar/
    ├── __init__.py
    ├── admin.py
    ├── apps.py
    ├── migrations/
    ├── models.py
    ├── tests.py
    └── views.py


# 1. Inicialize o repositório (se necessário)
git init

# 2. Adicione os arquivos e faça um commit
git add .
git commit -m "Initial commit"

# 3. Adicione o repositório remoto
git remote add origin https://github.com/welligtoncos/Aplicativos-IA.git

# 4. Envie o código para o GitHub (supondo que o branch seja 'main')
git branch -M main
git push -u origin main
>>>>>>> b119eed (Initial commit)


git ls-remote origin
sudo apt update
sudo apt install gh
gh auth login
gh repo list seu-usuario

pip freeze > requirements.txt
pip install -r requirements.txt
home/cloud/.pyenv/versions/3.10.4/bin/python /mnt/c/welligton-pos-IA/redeneural-matematica/redeneuralensinar/management/commands/respon
der_api.py