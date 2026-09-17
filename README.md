# Frontera Eficiente de Markowitz — Segmentos de Tarjeta de Crédito

Dashboard en **Streamlit** que aplica el modelo de optimización de portafolios
de **Markowitz** a los segmentos de tarjeta de crédito (TDC) del banco,
tratando cada segmento (Clásica, Oro, Platino, Black/Infinite, Empresarial,
Recompensas, Estudiantil) como una clase de activo.

## Contenido del repositorio

```
.
├── app.py                  # App de Streamlit (interfaz y visualizaciones)
├── markowitz_engine.py     # Lógica de optimización (independiente de Streamlit)
├── requirements.txt
├── .streamlit/config.toml  # Tema visual (azul / negro / blanco)
└── data/
    └── BD_Clientes_TDC_Markowitz_complementaria.xlsx   # Base de datos por defecto
```

## Datos de entrada

El archivo Excel debe contener (al menos) la hoja:

- **`Rendimientos_Mensuales_Segmento`**: columna `Fecha` (AAAA-MM) + una
  columna por segmento con el rendimiento neto mensual (fracción, ej. 0.021).

Opcionalmente, la hoja **`Clientes`** (con columnas `Segmento_Tarjeta` y
`Saldo_Actual_MXN`) habilita la pestaña de comparación contra la cartera
real del banco.

Desde la app se puede sustituir la base por defecto subiendo otro `.xlsx`
con la misma estructura, desde la barra lateral.

## Qué calcula

- Rendimiento y volatilidad anualizados por segmento.
- Matriz de covarianza y correlación entre segmentos.
- **Frontera eficiente** (optimización con `scipy.optimize.minimize`, SLSQP).
- Portafolio de **mínima varianza** y portafolio de **máximo Sharpe** (tangente).
- Nube de portafolios simulados (Monte Carlo) para referencia visual.
- Comparación de la **cartera actual** del banco (pesos por saldo vigente)
  contra la frontera eficiente.

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Subir a GitHub

```bash
git init
git add .
git commit -m "Modelo de frontera eficiente de Markowitz - segmentos TDC"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
git push -u origin main
```

## Desplegar en Streamlit Community Cloud

1. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
2. Clic en **New app**.
3. Selecciona el repositorio, la rama `main` y el archivo principal `app.py`.
4. Clic en **Deploy**. Streamlit instalará `requirements.txt` automáticamente.

## Estilo visual

Tema fintech/bursátil, definido en `.streamlit/config.toml` y con CSS
adicional en `app.py`:

- Paleta azul (`#0038A8` / `#3D8BFF`) y negro (`#0A0A0A`).
- Barra lateral con fondo negro y tipografía **Arial blanca**.
- Cuerpo principal con fondo blanco y tipografía **Arial negra**.
- Tarjetas de métricas y encabezado tipo "ticker" bursátil en negro con acentos azules.

## Notas metodológicas

- Los rendimientos se anualizan multiplicando la media mensual por 12; la
  covarianza mensual se escala también por 12 (aproximación estándar,
  no compuesta).
- Por defecto no se permiten ventas en corto (pesos entre 0% y 100%); esto
  es configurable desde la barra lateral.
- Este dashboard es una herramienta de apoyo a la decisión y no constituye
  una recomendación de inversión.
