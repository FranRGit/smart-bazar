# POS Inteligente: clasificación, recomendación y CRUD

## 1. Propósito y límite analítico

La vista **POS Inteligente** integra tres responsabilidades diferentes:

1. una terminal de venta simulada;
2. una predicción del método de pago mediante un clasificador entrenado;
3. un historial CRUD persistido en SQLite.

Solo la predicción **YAPE frente a EFECTIVO** usa un modelo de minería de datos en tiempo de ejecución. La recomendación de combo no ejecuta Apriori y el encabezado «K-Means Online» no corresponde a ningún algoritmo activo. El catálogo, los productos sugeridos y los valores de lift están escritos directamente en el código.

### Etiquetas de procedencia

| Etiqueta | Significado |
|---|---|
| **En vivo** | Resultado recalculado con las entradas actuales o con SQLite. |
| **Preentrenado** | Inferencia de un modelo aprendido anteriormente. |
| **Precalculado** | Número producido offline y leído como artefacto. |
| **Hardcodeado** | Regla, valor o catálogo escrito directamente en Python. |

## 2. Ruta activa, archivos y funciones

<code>app.py</code>, líneas 478-491, incluye la opción; las líneas 521-522 llaman al alias <code>show_pos()</code>, importado desde <code>src.panel_crud.show_panel</code>.

| Archivo | Influencia real |
|---|---|
| <code>app.py</code> | Enruta a la vista. |
| <code>src/panel_crud.py</code> | Terminal, construcción de características, inferencia, recomendaciones fijas, SQLite y CRUD. |
| <code>src/panel_predictivo.py</code> | <code>cargar_modelo()</code> reconstruye el encoder y carga el clasificador ganador. |
| <code>models/modelo_metodo_pago.json</code> | Metadatos: tipo XGBoost, columnas, clases de departamento y archivo del booster. |
| <code>models/modelo_metodo_pago_booster.json</code> | Árboles XGBoost preentrenados usados por <code>predict()</code> y <code>predict_proba()</code>. |
| <code>Notebooks/Panel_2_Prediccion_Metodo_Pago.ipynb</code> | Origen offline de características, entrenamiento, evaluación y exportación. No se ejecuta en el POS. |
| <code>db_consultas.sqlite</code> | Historial operativo local. No es el dataset de entrenamiento. |

Funciones principales:

| Función/constante | Papel |
|---|---|
| <code>init_db()</code> | Crea la tabla <code>consultas</code> si no existe. |
| <code>run_query()</code> | Ejecuta SELECT, INSERT, UPDATE o DELETE con parámetros SQLite. |
| <code>cargar_modelo()</code> | Carga el XGBoost actual desde formato nativo; está definida y cacheada en <code>panel_predictivo.py</code>. |
| <code>predict_payment_real()</code> | Construye una fila de siete variables y devuelve clase y confianza. |
| <code>mock_recommend_combo()</code> | Selecciona una recomendación desde un diccionario fijo. |
| <code>mock_predict_payment()</code> | Predictor por reglas que existe en el archivo, pero **no es llamado** por la ruta activa. |
| <code>PRODUCT_CATALOG</code> | Catálogo fijo que propone departamento y precio inicial. |
| <code>show_panel()</code> | Orquesta carga, terminal, inserción, métricas e interfaz CRUD. |

## 3. Flujo completo

Al abrir la opción ocurre lo siguiente:

1. <code>init_db()</code> garantiza la existencia de la tabla.
2. <code>cargar_modelo()</code> intenta cargar el artefacto **Preentrenado**.
3. Si carga, cada rerun construye una fila y ejecuta XGBoost.
4. <code>mock_recommend_combo()</code> obtiene una recomendación **Hardcodeada**.
5. Al confirmar, se guardan entradas, predicción y recomendación en SQLite.
6. La segunda pestaña consulta esa tabla y calcula indicadores **En vivo**.

No hay flujo desde el POS hacia <code>datasets/limpio</code>, inventario, notebook o reentrenamiento:

    entradas de pantalla
            |
            +--> XGBoost persistido --> pago predicho
            |
            +--> diccionario fijo ----> combo sugerido
            |
            +--> SQLite --------------> historial y KPIs

## 4. Clasificador de método de pago

### 4.1 Cómo se entrenó

El notebook construyó una fila por venta al unir ventas con su detalle. La variable objetivo fue:

\[
target =
\begin{cases}
1,& Metodo\_Pago=YAPE\\
0,& Metodo\_Pago=EFECTIVO
\end{cases}
\]

Se compararon Random Forest y XGBoost con un split estratificado 80/20. El XGBoost se configuró con 200 árboles, profundidad máxima 4, tasa de aprendizaje 0.05 y <code>scale_pos_weight</code> para compensar el desbalance. Fue exportado porque su F1 (0.4000) superó al de Random Forest (0.3866), aunque su exactitud fue 0.5851.

XGBoost construye árboles de decisión secuencialmente. Cada nuevo árbol intenta corregir el gradiente del error logístico de los anteriores:

\[
F_M(x)=\sum_{m=1}^{M}\eta f_m(x),
\qquad
P(YAPE\mid x)=\frac{1}{1+e^{-F_M(x)}}
\]

El POS **no entrena árboles**. <code>cargar_modelo()</code> lee el JSON de metadatos, reconstruye <code>LabelEncoder</code>, crea <code>XGBClassifier</code> y carga el booster nativo. El modelo se cachea con <code>st.cache_resource</code>.

### 4.2 Fila realmente enviada al modelo

<code>predict_payment_real()</code>, líneas 68-109, recibe <code>total</code>, <code>hora</code> y <code>departamento</code>, pero no todos influyen.

| Característica | Valor en el entrenamiento | Valor construido por POS | Procedencia/limitación |
|---|---|---|---|
| <code>Total</code> | Total real de la venta | Monto ingresado | **En vivo**, única medida transaccional real |
| <code>n_items</code> | Suma de cantidades del detalle | Siempre 2 | **Hardcodeado**, aproximación; el ticket visual muestra 1 unidad |
| <code>n_productos_distintos</code> | Conteo de IDs distintos | Siempre 1 | **Hardcodeado** |
| <code>departamento_principal_enc</code> | Moda del departamento codificada | Código del departamento elegido | **En vivo** si la clase existe |
| <code>pct_fotocopiadora</code> | Proporción de líneas de fotocopiadora | 1 solo para FOTOCOPIADORA; 0 en otro caso | Aproximación binaria |
| <code>dia_semana</code> | Día de la fecha real, lunes 0 a domingo 6 | Día actual del servidor | **En vivo**, pero no proviene de una fecha elegida |
| <code>es_fin_de_semana</code> | Indicador derivado de la fecha | 1 si el día actual es sábado/domingo | **En vivo** |

El artefacto actual solo conoce las clases <code>FOTOCOPIADORA</code> y <code>UTILES</code>. La interfaz ofrece además <code>GOLOSINAS</code>, <code>BEBIDAS</code> y <code>SERVICIOS</code>. Para una clase desconocida, <code>LabelEncoder.transform()</code> falla y el código asigna 0, el mismo código numérico de FOTOCOPIADORA. Sin embargo, <code>pct_fotocopiadora</code> queda en 0, formando una combinación que probablemente no estuvo representada en entrenamiento.

### 4.3 Entradas que no usa el modelo

| Entrada visible | Uso real |
|---|---|
| Hora del día | Se guarda en SQLite, pero <code>predict_payment_real()</code> nunca la incorpora en la fila. Mover el slider no cambia por sí solo la predicción. |
| Producto seleccionado | Solo propone precio y departamento iniciales; su nombre no llega al modelo. |
| ID de cliente | Solo se persiste. |
| Código de venta | Identificador de SQLite. |
| Precio del catálogo | Es el valor inicial de <code>Total</code>; el usuario puede reemplazarlo. |

La hora aparece en la firma de la función, lo que puede hacer creer que interviene, pero es un parámetro ignorado.

### 4.4 Clase, probabilidad y confianza

El clasificador devuelve:

- <code>pred = 1</code>: YAPE;
- <code>pred = 0</code>: EFECTIVO;
- <code>prob</code>: probabilidad estimada de YAPE;
- confianza mostrada: <code>prob</code> si gana YAPE, o <code>1-prob</code> si gana EFECTIVO.

Por tanto:

\[
confianza=\max(P(YAPE),P(EFECTIVO))
\]

Es confianza de la **clasificación de pago**, no confianza de la recomendación Apriori. Su ubicación dentro de la tarjeta «Reglas Apriori» puede inducir a confusión.

Si la carga del modelo falla, <code>show_panel()</code> muestra una advertencia y <code>predict_payment_real()</code> devuelve siempre **EFECTIVO, 50 %**. El mensaje habla de «simulador de contingencia», pero no llama a <code>mock_predict_payment()</code>.

## 5. Catálogo, ticket e impuestos

### 5.1 Catálogo fijo

<code>PRODUCT_CATALOG</code>, líneas 125-134, contiene ocho opciones:

| Producto | Departamento inicial | Precio inicial |
|---|---:|---:|
| Inca Kola 2L | BEBIDAS | S/ 10.00 |
| Cuaderno A4 College | UTILES | S/ 8.50 |
| Mica A4 Vinifan (10 und) | FOTOCOPIADORA | S/ 5.00 |
| Lapicero Pilot G2 | UTILES | S/ 6.00 |
| Impresión B/N (20 copias) | FOTOCOPIADORA | S/ 4.00 |
| Galletas Oreo / Soda | GOLOSINAS | S/ 3.50 |
| Folder Manila A4 (Paquete) | SERVICIOS | S/ 7.00 |
| Entrada libre / monto personalizado | UTILES | S/ 12.50 |

Estos datos son **Hardcodeados**; no se consultan desde <code>inventario.csv</code>. El producto tampoco se guarda en la tabla, por lo que el historial no permite reconstruir qué se vendió.

El código de venta inicial es aleatorio entre 100000 y 999998 y el cliente inicial entre 1 y 149. Son valores de conveniencia, no predicciones.

### 5.2 Ticket virtual

La pantalla muestra una sola unidad y usa el monto introducido como total final. La sugerencia se presenta como upsell, pero:

- no se añade al ticket;
- no modifica el total;
- no descuenta inventario;
- no crea una segunda línea;
- el texto «Anular / Limpiar Item» no tiene una acción conectada;
- «Último escaneo» es simplemente la selección actual, no un historial.

### 5.3 Extracción del IGV

El monto se interpreta como precio **incluido IGV**:

\[
Subtotal=\frac{Total}{1.18}
\]

\[
IGV=Total-Subtotal=\frac{18}{118}Total
\]

\[
Total\ a\ pagar=Total
\]

Este cálculo es correcto para extraer un impuesto de 18 % incluido en el precio. No sería la fórmula para agregar IGV a un subtotal. Subtotal e IGV solo se muestran; SQLite almacena únicamente el total.

## 6. Recomendación de combo

<code>mock_recommend_combo()</code>, líneas 112-121, es una tabla de decisión:

| Departamento | Condición | Resultado fijo |
|---|---|---|
| FOTOCOPIADORA | cualquiera | MICA A4 VINIFAN, lift 3.21 |
| UTILES | total menor o igual a 15 | LAPICERO PILOT, lift 2.89 |
| UTILES | total mayor a 15 | CAJA COLORES FABER CASTELL, lift 1.80 |
| GOLOSINAS | cualquiera | GALLETAS OREO, lift 1.95 |
| BEBIDAS | cualquiera | GALLETAS SODA, lift 1.72 |
| SERVICIOS | cualquiera | FOLDER MANILA A4, lift 2.54 |
| Otro valor | fallback | MICA A4 VINIFAN, lift 2.10 |

En reglas de asociación, el lift verdadero sería:

\[
lift(A\rightarrow B)=\frac{support(A\cup B)}{support(A)\,support(B)}
\]

Un lift mayor que 1 indica coocurrencia superior a la esperada bajo independencia. Sin embargo, en POS:

- no se construye una canasta;
- no se invoca Apriori;
- no se leen reglas desde un artefacto;
- no se recalculan soporte, confianza ni lift;
- los lifts son texto fijo sin trazabilidad activa al dataset.

Por ello, la recomendación debe presentarse como **regla comercial hardcodeada inspirada en resultados de asociación**, no como Apriori online.

El encabezado también afirma «K-Means Online», pero <code>panel_crud.py</code> no importa ni ejecuta K-Means y no asigna segmentos.

## 7. Persistencia SQLite

### 7.1 Esquema

<code>init_db()</code>, líneas 17-34, crea:

| Columna | Tipo/restricción | Contenido |
|---|---|---|
| <code>id</code> | INTEGER PK AUTOINCREMENT | Identificador interno |
| <code>id_venta</code> | TEXT UNIQUE | Código ingresado y convertido a mayúsculas |
| <code>id_cliente</code> | INTEGER | Cliente ingresado |
| <code>total</code> | REAL | Total incluido IGV |
| <code>hora</code> | INTEGER | Slider de 8 a 22 |
| <code>departamento</code> | TEXT | Selección del POS |
| <code>metodo_pago_predicho</code> | TEXT | YAPE o EFECTIVO predicho, no observado |
| <code>recomendacion_combo</code> | TEXT | Texto fijo elegido, incluido el lift |
| <code>timestamp</code> | TEXT | Fecha/hora del servidor |

La ruta <code>db_consultas.sqlite</code> es relativa al directorio desde el cual se inicia Streamlit, a diferencia de las rutas de modelos ancladas al repositorio. Ejecutar la aplicación desde otro directorio puede crear o leer otra base con el mismo nombre.

<code>run_query()</code> abre una conexión por operación. Usa placeholders <code>?</code>, lo que evita concatenar directamente entradas del usuario en SQL. En escrituras hace commit y devuelve <code>lastrowid</code>; en lecturas devuelve filas y nombres de columnas.

### 7.2 Crear

El botón valida que <code>id_venta</code> no esté vacío, genera timestamp e inserta los ocho valores. La unicidad se controla por SQLite; un código repetido produce <code>IntegrityError</code>.

«Confirmar pago» no registra el método realmente elegido por el cliente. Guarda la predicción como si fuera atributo de la transacción. Tampoco se persisten producto, cantidad, confianza, subtotal ni IGV.

### 7.3 Leer

El historial ejecuta un SELECT ordenado por <code>id DESC</code>. La tabla mostrada es una copia **En vivo** del contenido SQLite al producirse el rerun. Si no hay filas, solo aparece un mensaje.

### 7.4 Actualizar

El formulario permite cambiar cliente, total, hora, departamento y método «predicho». También reemplaza el timestamp por la hora de edición.

No vuelve a llamar al clasificador ni a la recomendación. Por ejemplo, una venta puede cambiar de FOTOCOPIADORA a BEBIDAS y conservar «MICA A4 VINIFAN (Lift: 3.21)». El usuario también puede elegir manualmente YAPE/EFECTIVO, por lo que después de una edición el campo ya no necesariamente representa la salida del modelo.

El código de venta y la recomendación no son editables. No existe fecha de creación separada de fecha de actualización.

### 7.5 Eliminar

DELETE borra físicamente la fila seleccionada y ejecuta <code>st.rerun()</code>. La interfaz avisa que es irreversible; no hay papelera, auditoría ni borrado lógico.

## 8. Indicadores del historial

Todos se calculan **En vivo desde SQLite**, no desde <code>ventas.csv</code>:

| Indicador | Fórmula | Interpretación correcta |
|---|---|---|
| Total Ventas | \(N=\text{número de filas}\) | Registros guardados en esta base local |
| Ingreso Acumulado | \(\sum_{i=1}^{N} total_i\) | Suma de montos persistidos |
| Ticket Promedio | \(\frac{1}{N}\sum total_i\) | Media aritmética por registro |
| Adopción Yape | \(100\frac{\#(metodo\_pago\_predicho=YAPE)}{N}\) | Proporción de etiquetas almacenadas como YAPE |

«Adopción Yape» no mide pagos reales. Mide predicciones guardadas y posibles ediciones manuales. El nombre correcto desde el punto de vista analítico sería «porcentaje de registros clasificados como YAPE».

Eliminar o editar filas altera inmediatamente estos KPIs. Registrar ventas en el POS no altera los indicadores de EDA, clustering o predicciones, porque aquellos consumen los CSV, no esta base.

## 9. Inventario de resultados visibles

| Resultado/control | Algoritmo/fuente | Tipo |
|---|---|---|
| Método de pago previsto | <code>XGBClassifier.predict()</code> | **Preentrenado + En vivo** |
| Confianza | Probabilidad de la clase ganadora | **Preentrenado + En vivo** |
| Combo y lift | Diccionario y condición de monto | **Hardcodeado** |
| Departamento/precio inicial | <code>PRODUCT_CATALOG</code> | **Hardcodeado** |
| Subtotal, IGV y total | Aritmética tributaria | **En vivo** |
| Ticket visual | Entradas actuales; una unidad fija | **En vivo/Hardcodeado** |
| «Último escaneo» | Producto actualmente seleccionado | **UI**, no histórico |
| «K-Means Online» | Sin implementación | Texto **Hardcodeado** |
| Tabla de historial | SELECT de SQLite | **En vivo** |
| Cuatro KPIs | Agregaciones de SQLite | **En vivo** |

## 10. Fallos, inconsistencias y límites

- Un fallo de modelo produce EFECTIVO al 50 %, no la lógica de <code>mock_predict_payment()</code>.
- La hora se almacena y aparece en formularios, pero no influye en XGBoost.
- Tres de los cinco departamentos de la UI son desconocidos para el encoder entrenado.
- El POS supone 2 ítems para el modelo mientras muestra una unidad en el ticket.
- La recomendación y sus lifts no son resultados de Apriori en línea.
- No existe K-Means en esta ruta.
- El producto no se persiste y no se puede auditar el artículo vendido.
- No se registra el pago observado, por lo que no se puede medir acierto productivo ni realimentar el modelo.
- Las ediciones no recalculan clase ni combo y pueden dejar campos contradictorios.
- SQLite no actualiza ventas, inventario, stock, modelos ni datasets analíticos.
- No hay transacción conjunta con inventario ni control de concurrencia de caja; SQLite puede bloquearse con escrituras simultáneas.
- Un cambio de versión XGBoost genera advertencia, pero la carga continúa porque se usa el formato nativo.
- Si los metadatos declaran columnas distintas a las que construye POS, la selección <code>DataFrame[columnas]</code> puede fallar.

## 11. Guion para exposición

1. **Separar responsabilidades:** POS/CRUD no es lo mismo que minería de datos.
2. **Explicar el entrenamiento:** XGBoost aprendió offline patrones de siete variables para clasificar YAPE o EFECTIVO.
3. **Explicar inferencia:** la pantalla arma una sola fila, ordena sus columnas como exige el artefacto y obtiene clase y probabilidad.
4. **Ser transparente con aproximaciones:** dos ítems, un producto distinto y el día actual sustituyen información que el POS no captura.
5. **Distinguir la recomendación:** el lift es un valor fijo de catálogo; no se ejecuta Apriori.
6. **Explicar persistencia:** SQLite guarda una consulta/venta local y permite crear, leer, editar y eliminar.
7. **Interpretar KPIs correctamente:** resumen la base local y «Adopción Yape» representa etiquetas predichas, no pagos confirmados.
8. **Cerrar con la limitación principal:** sin guardar el resultado real del pago no puede evaluarse el modelo en producción ni aprender de nuevas ventas.

