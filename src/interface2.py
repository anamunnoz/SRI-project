import streamlit as st

# Definir credenciales de usuarios
users = {
    "usuario1": "password1",
    "usuario2": "password2",
    "admin": "admin123"
}

# Función de autenticación
def autenticar(username, password):
    if username in users and users[username] == password:
        return True
    else:
        return False

# Interfaz de usuario para el login
def login_page():
    st.title("Inicio de Sesión")

    # Formulario de inicio de sesión
    username = st.text_input("Nombre de Usuario")
    password = st.text_input("Contraseña", type="password")
    login_button = st.button("Iniciar Sesión")

    if login_button:
        if autenticar(username, password):
            st.success(f"Bienvenido, {username}!")
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
        else:
            st.error("Nombre de usuario o contraseña incorrectos")

# Mostrar el contenido principal si está logeado
def main_page():
    st.title("Página Principal")
    st.write(f"¡Has iniciado sesión como {st.session_state['username']}!")

    logout_button = st.button("Cerrar Sesión")
    if logout_button:
        st.session_state['logged_in'] = False

# Gestión del estado de sesión
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main_page()
else:
    login_page()
