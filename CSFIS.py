import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, classification_report


preguntas = [
    {
        'pregunta': 'En la organizacion , como calificarias la implementacion de Politicas de Seguridad de la Informacion (Reglas , leyes , normas)  ',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'En la organizacion , como calificarias la implementacion de estandares de Seguridad de la informacion como la ISO 27001 , NIST , etc',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'En la organizacion , como calificarias la implementacion de documentos para gestionar la seguridad de la informacion (Manuales , Portafolios , Leyes impresas)',
        'opciones': [1,2,3,4,5]
    },
     {
        'pregunta': 'Respecto a la seguridad fisica , como calificarias la implementacion de sistemas de deteccion de intrusos o control de entrada',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'Respecto a la seguridad de programas , como calificarias la implementacion de gestion de accesos como roles separados , privilegios por rol , control de perfiles',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'Respecto a la gestion de riesgos: como calificarias la implementacion de procesos que identifiquen vulnerabilidades como auditorias , evaluaciones , mantenimiento',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'Respecto a la gestion de riesgos , como calificarias la gestion de incidentes de seguridad de la informacion como procesos a seguir , medidas de mitigacion , identificacion',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a la gestion de rendimiento , como calificarias la implementacion de procesos o areas de control de presupuesto',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a la gestion de rendimiento , como calificarias la implementacion de procesos de provision de recursos de TI , herramientas , equipos de trabajo , etc',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a la gestion de rendimiento , como calificarias la implementacion de auditorias de seguridad de la informacion',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'Respecto al desarrollo de competencias , como calificarias la implementacion de programas de concientizacion en temas de seguridad de la informacion',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'Respecto al desarrollo de competencias , como calificarias la implementacion de programas que permitan desarrollar las capacidades tecnologicas en la organizacion',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'Respecto a la continuidad del negocio , como calificarias la implementacion de un plan o estrategia de continuidad del negocio',
        'opciones': [1,2,3,4,5]
    },
    {
        'pregunta': 'Respecto a la continuidad del negocio , como calificarias la implementacion de marcos de gestion o continuidad basada en TI ',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a la colaboracion con externos , como calificarias la implementacion de procesos que aseguren la cooperacion de estos como contrato , concientizacion , etc',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto al conocimiento del equipo , como calificarias el nivel de habilidad en manejo de TI de los empleados',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto al conocimiento del equipo , como calificarias la conducta que tienen los empleados respecto a los procesos o medidas de seguridad de la informacion',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto al conocimiento del equipo , como calificarias el compromiso de la alta gerencia al respecto de implementar medidas de seguridad de la informacion',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto al compromiso en seguridad de la informacion , como calificarias la implementacion de medidas para regular la responsabilidad del staff',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto al compromiso en seguridad de la informacion , como calificarias la implementacion de procesos que desarrollen valores de cooperacion en la organizacion',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a la cultura en seguridad de la informacion , como calificarias tu el liderazgo y compromiso de la alta gerencia respecto a ello',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a la cultura en seguridad de la informacion , como calificarias la provision de recursos al equipo para asegurar la SI',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a la cultura en seguridad de la informacion , como calificarias la comunicacion que tiene la alta gerencia con el equipo ',
        'opciones': [1, 2, 3, 4, 5]
    },
        {
        'pregunta': 'Respecto a las prioridades y estructura de la organizacion , como calificarias la implementacion de practicas de trabajo en SI',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a las prioridades y estructura de la organizacion ,  como calificarias el cumplimiento de leyes nacionales respecto a seguridad de la organizacion',
        'opciones': [1, 2, 3, 4, 5]
    },
    {
        'pregunta': 'Respecto a las prioridades y estructura de la organizacion ,  como calificarias la orientacion que tiene la organizacion respecto a la seguridad de la informacion ',
        'opciones': [1, 2, 3, 4, 5]
    }
]

def mostrar_pregunta(pregunta):
    st.write(f"**Pregunta:** {pregunta['pregunta']}")
    respuesta = st.radio("Opciones:", pregunta['opciones'])
    return respuesta

def load_data():
    data = pd.read_csv('CSFIS_DEF.csv')
    return data

def division_datos(data):
    x = data.iloc[:, 4:15]
    y = data.iloc[:, 15]

    X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=21)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test , scaler


def rf_model(x,y):
    
    n_estimators = [25,50,100]
    max_depth = [10,20,100,None]
    min_samples_leaf = [1,2,4,10]
    bootstrap = [True, False]

    param_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
        "criterion": ['gini','entropy']
    }

    rf = RandomForestClassifier(random_state=42)
    gs = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                       n_iter=10, cv=2,random_state=42, scoring='accuracy')
    gs.fit(x,y)

    return gs

def formulario():
    form = st.form(key="CSF")
    form.header("Formulario")
    
    respuestas = []

    for i, pregunta in enumerate(preguntas):
        form.write(f"**Pregunta {i+1}:** {pregunta['pregunta']}")
        respuesta = form.radio(f"Opciones para pregunta {i+1}", pregunta['opciones'])
        respuestas.append(respuesta)

    form.form_submit_button("Completado")

    return respuestas

def crear_input_data(rpt):
    input_data ={
        "IP" :[ rpt[0] + rpt[1] + rpt[2]],
        "ISF" : [rpt[3] + rpt[4]], 
        "GR" : [rpt[5] + rpt[6]],
        "GRE" : [rpt[7] + rpt[8] + rpt[9]],
        "DC" : [rpt[10] + rpt[11]], 
        "CN" : [rpt[12] + rpt[13]],
        "EXT" : [rpt[14]],
        "CE": [rpt[15] + rpt[16] + rpt[17]] ,
        "CSI": [rpt[18] + rpt[19]],
        "CU" : [rpt[20] + rpt[21] + rpt[22]],
        "PRI" : [rpt[23] + rpt[24] + rpt[25]]
    }

    return pd.DataFrame.from_dict(input_data)

def main():
    data = load_data()
    X_train, X_test, y_train, y_test , scaler = division_datos(data)
    modelo = rf_model(X_train,y_train)

    y_pred = modelo.best_estimator_.predict(X_test)

    rpt = formulario()

    input = crear_input_data(rpt)

    input = scaler.transform(input)

    mejor_modelo = modelo.best_estimator_

    prediccion = mejor_modelo.predict(input)

    st.write(input)
    st.write(prediccion[0])

if __name__ == "__main__":
    st.title('Calculo de exito de implementacion')
    main()

    