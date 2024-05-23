import streamlit as st
import psycopg2
from chatbot import predict_class, get_response, intents
import nltk
nltk.download('punkt')
nltk.download('wordnet')



import os

my_database = os.environ.get('my_database')
my_host = os.environ.get('my_host')
my_password = os.environ.get('my_password')
my_port = os.environ.get('my_port')
my_user = os.environ.get('my_user')

# Configura las variables de conexión
HOST = my_host
DATABASE = my_database
USER = my_user
PASSWORD = my_password
PORT = my_port  # Asegúrate de que este valor sea un entero

# Configura la conexión a la base de datos
def get_db_connection():
    return psycopg2.connect(
        host=HOST,
        database=DATABASE,
        user=USER,
        password=PASSWORD,
        port=PORT
    )

# Función para crear la tabla si no existe
def create_table_if_not_exists():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            pregunta TEXT,
            respuesta TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

# Función para guardar una pregunta y su respuesta en la base de datos
def save_message_to_db(pregunta, respuesta):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (pregunta, respuesta) VALUES (%s, %s)", (pregunta, respuesta))
    conn.commit()
    cursor.close()
    conn.close()

# Crear la tabla si no existe
create_table_if_not_exists()

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="Chaty", page_icon="img/cropped-Beyond-Education_Horizonatal-color.png")
st.title(":male_mage: Asistente virtual:robot_face:")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "first_message" not in st.session_state:
    st.session_state.first_message = True

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿Como puedo ayudarte?")

    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿Como puedo ayudarte?"})

    st.session_state.first_message = False

if prompt := st.chat_input("cómo puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Implementación del algoritmo
    insts = predict_class(prompt)
    res = get_response(insts, intents)

    with st.chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})

    # Guardar pregunta y respuesta en la base de datos
    save_message_to_db(prompt, res)
