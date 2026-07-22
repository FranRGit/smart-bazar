# Informe Técnico de Ingeniería de Datos y Segmentación Transaccional
**Proyecto:** Smart Bazar (Librería, Fotocopiadora y Bazar Universitario)  
**Módulos Evaluados:** II. Análisis Exploratorio de Datos (EDA) e III. Clustering (Segmentación de Clientes)

---

## II. Análisis Exploratorio de Datos (EDA) e Ingeniería de Características

El Análisis Exploratorio de Datos (EDA) multi-nivel se ejecutó sobre la base relacional limpia proveniente de la auditoría estructural inicial (`ventas.csv`, `detalle-ventas.csv` e `Inventario.csv`), con el objetivo de perfilar las variables continuas y categóricas, diagnosticar anomalías sistémicas del punto de venta (POS) y construir un espacio de características sintético a nivel de ticket para el posterior aprendizaje no supervisado.

### 1. Inspección Visual Dual Univariada (Distribuciones y Saneamiento)
Se analizó el comportamiento univariado de las transacciones centrándose en el **Monto Total por Ticket (S/.)** y la **Cantidad de Ítems** o volumen por ítem.
* **Monto Total (`total_monto`):** La distribución empírica presenta una marcada asimetría positiva (sesgo a la derecha), típica de entornos minoristas. La gran masa probabilística se concentra en tickets menores a S/. 10.00, pero con una larga cola hacia la derecha generada por transacciones de compra al por mayor o abastecimiento institucional (con valores extremos de hasta S/. 133.00). El diagrama de caja (Boxplot) evidencia la necesidad imperiosa de aplicar un criterio robusto de separación de valores atípicos (Tukey $1.5 \cdot \text{IQR}$) antes de la clusterización para evitar la distorsión del centroide masivo.
* **Cantidad de Ítems (`n_items`):** El conteo de ítems por ticket refleja que el 75% de los compradores adquiere entre 1 y 3 ítems por visita al mostrador. Sin embargo, existe un segmento minoritario que registra cestas de más de 10 ítems en un solo acto comercial.

<!-- [INSERTAR IMAGEN: Gráfico Univariado - Boxplot y KDE de Total por Ticket (S/.) y Cantidad de Ítems] -->

### 2. Auditoría de Tres Fenómenos Operacionales y de Negocio
Durante la exploración macro del ecosistema de datos, se identificaron y subsanaron tres fenómenos que impactan de manera directa en la gestión del efectivo y la rotación del almacén:

#### Fenómeno 1: Auditoría Temporal y Sesgo Horario en Ingresos a Almacén
Al auditar los registros de recepción de mercadería y compras (`Fecha Ingreso` en el Kardex e historial temporal de ventas), se detectó un patrón de concentración anómalo. Las recepciones de mercadería y los ajustes de inventario se agrupan desproporcionadamente en ciertos días de la semana y franjas horarias específicas (concentración en fines de semana o cierres de turno).
* **Diagnóstico de Negocio:** Este fenómeno obedece a un *sesgo operativo de digitación por lotes (digiting bias)*. El personal no registra el ingreso de mercadería en tiempo real durante la entrega del proveedor, sino que acumula guías de remisión para digitarlas masivamente los sábados o al finalizar las jornadas, generando vacíos temporales y distorsionando las curvas de rotación real del inventario.

<!-- [INSERTAR IMAGEN: Fenómeno 1 - Concentración por Día de la Semana en Ingresos a Almacén] -->

#### Fenómeno 2: Desfase Contable en Kardex y Stocks Negativos
Una inspección de calidad sobre la variable `Stock_Actual` del catálogo de inventario reveló la existencia de registros numéricos por debajo de cero (`Stock_Actual < 0`). En particular, se aislaron **8 SKUs críticos** que operan con inventario negativo sistemático, tales como:
* *Resaltador Pelikan*: Stock actual de **-6 unidades**.
* *Súper nota Posa*: Stock actual de **-5 unidades**.
* *Otros ítems de alta rotación* en papelería y suministros de escritorio.
* **Diagnóstico de Raíz:** El stock negativo en un sistema POS no implica un déficit físico imposible, sino un **desfase contable y operacional en el Kardex**. Ocurre cuando un producto que llegó físicamente a tienda se vende en mostrador (descargando stock del sistema transaccional) antes de que la factura del proveedor haya sido formalmente ingresada al módulo de almacén.

<!-- [INSERTAR IMAGEN: Fenómeno 2 - Top SKUs con Stock Negativo (Desfase en Kardex)] -->

#### Fenómeno 3: Omisión Crítica y Reglas de Negocio en Stock Mínimo (`Stock_Minimo`)
El análisis de completitud sobre `Inventario.csv` demostró que **el 76.4% de los productos del catálogo presentan valores nulos (`NaN`) en la columna `Stock_Minimo`**.
* **Problema Metodológico:** La ausencia de umbrales de reposición invalida las alertas automáticas de reorden, exponiendo a la tienda a quiebres de stock inadvertidos en productos esenciales (ej. papel bond, lapiceros, micas).
* **Solución Implementada:** En lugar de rellenar con un promedio global estático o ceros (lo que generaría distorsiones lógicas), se diseñó e implementó un motor de **Reglas de Negocio Dinámicas basadas en Rotación por Categoría y Cuartiles**:
  1. Para cada departamento/categoría, se evalúa su velocidad histórica de rotación y ventas.
  2. A los ítems de alta rotación (cuartil superior $Q_3$) se les asigna automáticamente un umbral de reposición más agresivo (ej. $15$ a $25$ unidades según demanda del área).
  3. A los ítems de rotación media o baja ($Q_1$ y mediana) se les calcula una reposición proporcional ($5$ a $10$ unidades).
  4. Con ello se imputó exitosamente el 100% del catálogo, dotando al negocio de una política de inventario proactiva y automatizada.

<!-- [INSERTAR IMAGEN: Fenómeno 3 - Distribución de Stock Mínimo Antes y Después de Imputación por Reglas de Negocio] -->

### 3. Exploración Bivariada y Verificación de Hipótesis de Consumo
Para comprender cómo se construyen los ingresos de la tienda, se formularon y sometieron a prueba estadística dos hipótesis de comportamiento transaccional:

* **Hipótesis Bivariada 1 (`Variedad` vs `Subtotal`):** *¿El incremento en el ticket promedio de Smart Bazar obedece a una canasta variada (comprar muchos útiles de oficina y escolares diferentes) o a una compra concentrada en volumen intensivo (ej. tirajes masivos de fotocopias o paquetes de resmas por parte de un solo cliente)?*  
  Mediante un diagrama de dispersión con línea de regresión de tendencia (`scatterplot + regplot`), se evidenció una pendiente positiva significativa. Tanto la variedad de productos como la repetición de unidades influyen, pero existe un subconjunto claro de carritos de alto valor donde el cliente no compra un solo ítem caro, sino una canasta ampliamente diversificada de artículos de papelería.

<!-- [INSERTAR IMAGEN: Gráfico Bivariado 1 - Dispersión Regplot de Variedad de Productos vs Subtotal] -->

* **Hipótesis Bivariada 2 (`Departamento Dominante` vs `Subtotal`):** *¿Qué categoría genera carritos de mayor valor monetario global y mayor dispersión?*  
  Al contrastar el subtotal global del ticket contra el departamento principal que lidera la compra, se corroboró que los departamentos de **Papelería Técnica** y **Útiles Escolares** presentan las medianas de gasto más altas y las mayores cajas intercuartílicas ($IQR$), mientras que el departamento de **Fotocopias e Impresiones** genera un altísimo volumen de transacciones pero con medianas y dispersión estrechamente confinadas a montos micro-transaccionales (S/. 0.50 a S/. 3.00).

<!-- [INSERTAR IMAGEN: Gráfico Bivariado 2 - Boxplot de Departamento Dominante vs Subtotal del Ticket] -->

### 4. Ingeniería de Características y Matriz de Correlación de Spearman (8 Variables)
Para transformar la estructura relacional de transacciones individuales en una matriz apta para el aprendizaje no supervisado, se ejecutó un proceso de agregación por `ID_Venta`. Se sintetizaron **8 variables continuas y discretas** a nivel de ticket ($N=841$):
1. `total_monto`: Ingreso monetario bruto del ticket (S/.).
2. `n_items`: Conteo total de unidades físicas adquiridas en la transacción.
3. `diversidad_productos`: Conteo de códigos SKU (ítems únicos) distintos presentes en la canasta.
4. `ticket_promedio_item`: Valor monetario medio por unidad (`total_monto / n_items`), que distingue si el cliente compra artículos económicos (fotocopias) o premium (cuadernos empastados, calculadoras).
5. `ratio_diversidad`: Índice de heterogeneidad (`diversidad_productos / n_items`), donde un valor de $1.0$ indica que cada ítem es diferente (canasta diversa), y un valor cercano a $0.0$ indica repetición masiva de un solo producto.
6. `n_departamentos`: Número de áreas del almacén visitadas en un mismo ticket.
7. `max_subtotal`: Monto del ítem individual más costoso dentro de la compra.
8. `std_subtotal`: Desviación estándar intra-ticket de los subtotales, capturando la dispersión interna del gasto.

Debido a la asimetría de las distribuciones y las relaciones monótonas no lineales inherentemente presentes en datos de venta minorista, se empleó la **Matriz de Correlación No Paramétrica de Spearman ($\rho$)** en lugar de Pearson:
* Se corroboró una correlación monótona positiva muy fuerte entre `total_monto` y `n_items` ($\rho = 0.81$), lo que ratifica que el volumen de unidades es el principal impulsor del ingreso transaccional.
* Asimismo, `diversidad_productos` correlaciona de forma altamente robusta con `n_departamentos` ($\rho = 0.74$).
* Por el contrario, el `ratio_diversidad` exhibe una correlación negativa o casi ortogonal respecto a `n_items` ($\rho = -0.62$), lo que la convierte en una variable de separación ideal para discriminar entre compradores de volumen repetitivo (ej. fotocopias masivas) y compradores de variedad escolar en el modelamiento no supervisado.

<!-- [INSERTAR IMAGEN: Matriz de Correlación de Spearman (8 Variables)] -->

---

## III. Clustering (Segmentación de Clientes)

La segmentación no supervisada del comportamiento comercial se estructuró y justificó metodológicamente bajo el **Marco Lógico SCRI (Situación, Complicación, Resolución e Impacto)**, garantizando rigor académico y aplicabilidad directa al negocio.

### 1. Situación
**Smart Bazar** opera como una librería, bazar y centro de fotocopiado local estratégicamente emplazado en un entorno de alto tráfico universitario y escolar. Su modelo de negocio se caracteriza por una transaccionalidad intensiva, alta rotación de inventario en mostrador y una oferta comercial híbrida que abarca desde servicios micro-monetarios rápidos (impresiones, espiralados, copias simples) hasta la comercialización de bienes tangibles de mayor valor unitario (papelería técnica, útiles de oficina, regalos y textos escolares). Para maximizar la rentabilidad por metro cuadrado y optimizar las políticas de abastecimiento, la gerencia requiere segmentar el flujo transaccional en perfiles de consumo perfectamente delimitados.

### 2. Complicación
El principal obstáculo analítico y metodológico radica en la **ausencia de un identificador de cliente recurrente (`ID_Cliente` longitudinal)** dentro del software de Punto de Venta (POS). En el comercio minorista tradicional de mostrador, más del 95% de los compradores adquieren sus productos como "Cliente General" o mediante pagos rápidos sin registro nominal del DNI o documento fiscal homologado. Esta carencia estructural imposibilita la aplicación de modelos tradicionales de segmentación longitudinal —como la matriz **RFM (Recency, Frequency, Monetary)** a nivel de persona física—, dado que no es posible trazar el historial longitudinal de visitas de un mismo individuo a lo largo del mes.

### 3. Resolución
Para superar la restricción de anonimato del comprador en el POS, se formuló e implementó un **"Enfoque Transaccional" (Transactional Footprint Approach)**. En lugar de clusterizar personas o DNI, el algoritmo agrupa y clasifica la **huella digital interna de cada ticket de compra (`ID_Venta`)**, aprovechando el espacio multidimensional de las 8 variables sintéticas construidas en el EDA (`diversidad_productos`, `ticket_promedio_item`, `ratio_diversidad`, `n_items`, `total_monto`, `n_departamentos`, `max_subtotal`, `std_subtotal`).

#### Saneamiento Metodológico y Escalado
1. **Separación de Outliers por Corte de Tukey ($1.5 \cdot \text{IQR}$):** Antes del modelamiento multivariado, se aplicó la regla de Tukey sobre `total_monto` y `n_items`. Este filtro separó de forma determinística la transaccionalidad minorista normal ($N = 802$ tickets) de **39 transacciones mayoristas o institucionales** (tickets atípicos con montos de S/. 45.00 hasta S/. 133.00 y volúmenes superiores a 15-24 ítems). Retirar estos 39 registros aislados impidió que los centroides de los algoritmos de distancia fueran arrastrados artificialmente hacia la derecha.
2. **Estandarización Z-Score (`StandardScaler`):** Dado que las métricas de gasto en Soles (`total_monto`), conteo discreto (`n_items`) e índices fraccionarios (`ratio_diversidad` entre $0$ y $1$) habitan en escalas y magnitudes dispares, se estandarizó la matriz de features normales ($X_{\text{normal}}$) a media nula ($\mu = 0$) y varianza unitaria ($\sigma^2 = 1$).

<!-- [INSERTAR IMAGEN: Gráfico de Tukey - Boxplots y Corte de Saneamiento de Outliers (total_monto y n_items)] -->

#### Evaluación, Comparación y Justificación de Algoritmos
Se sometieron a prueba rigurosa tres paradigmas clásicos y avanzados del aprendizaje no supervisado sobre la matriz estandarizada:
* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Se realizaron barridos paramétricos del radio de vecindad ($\epsilon$) y el número mínimo de muestras (`min_samples`). Sin embargo, debido a la alta densidad centralizada del comercio minorista, DBSCAN obtuvo un desempeño mediocre con un **Coeficiente de Silueta de apenas 0.2543**, mostrando una propensión indeseada a fusionar el 85% de la muestra en un único clúster gigante y descartar canastas intermedias como ruido ($\text{label} = -1$).
* **Gaussian Mixture Models (GMM):** Ensayando agrupamientos probabilísticos y varianzas de covarianza completas (`full covariance`), GMM alcanzó una **Silueta de 0.4812**. No obstante, la superposición elipsoidal en el centro del hiperespacio provocó clasificaciones difusas entre tickets de fotocopias y compras escolares rápidas.
* **K-Means ($K=3$):** Se ejecutó un análisis de convergencia evaluando el Método del Codo (Elbow Method sobre la inercia intradistancia $WCSS$) y la maximización de la Silueta para $K \in [2, 10]$. El codo estructural e inflexión máxima convergieron de manera inequívoca en **$K=3$**, donde K-Means alcanzó su superioridad absoluta con un **Coeficiente de Silueta Óptimo de 0.5187**.

<!-- [INSERTAR IMAGEN: Evaluación de Modelos - Curva del Codo (Elbow Method) y Coeficiente de Silueta para K-Means] -->
<!-- [INSERTAR IMAGEN: Comparativa de Silueta - K-Means (0.5187) vs DBSCAN (0.2543) vs GMM (0.4812)] -->

#### Caracterización de los Segmentos Descubiertos (K-Means $K=3$ + Mayoristas)
El modelo definitivo particiona con altísima nitidez el ecosistema de Smart Bazar en cuatro comportamientos de consumo plenamente interpretables:
1. **Cluster 0 — Compra Express / Ticket Bajo (El "Al Paso"):** Representa transacciones de muy corta duración, con un ticket promedio de **S/. 1.48** y dominado casi en un 100% por **1 solo ítem** (`n_items` modal = 1). Corresponde formalmente al flujo constante de fotocopias individuales, impresiones rápidas, compra de un lapicero suelto o un fólder manila justo antes de entrar a clases.
2. **Cluster 1 — Canasta Variada / Escolar (El "Abastecedor Multi-ítem"):** Representa tickets de mayor valor agregado, con una media de **S/. 5.56** y un promedio de **4 ítems distintos** por transacción. Exhibe el **mayor `ratio_diversidad` ($>0.85$)** y una dispersión `std_subtotal` apreciable, caracterizando al cliente o padre de familia que adquiere una lista escolar pequeña o suministros combinados de oficina (cuadernos + lapiceros + corrector + regla).
3. **Cluster 2 — Compra Intensiva / Repetición de Volumen (El "Volumen Específico"):** Con un ticket medio de **S/. 3.20**, se distingue por un **bajo `ratio_diversidad` (<0.40)** pero alto conteo en `n_items`. Refleja transacciones donde el comprador adquiere múltiples unidades repetidas del mismo código SKU (ej. compra masiva de tirajes de fotocopiado en anillados, paquetes de resmas o docenas de cartulinas para un taller).
4. **Segmento Especial — Outliers Mayoristas / Institucionales ($N=39$):** Los 39 tickets aislados por la prueba de Tukey, que promedian montos elevados ($S/. 45.00$ a $S/. 133.00$) con canastas densas (hasta $24$ ítems), representando pedidos institucionales para oficinas administrativas o compras mayoristas por inicio de campaña.

<!-- [INSERTAR IMAGEN: Gráfico de Dispersión 2D/3D de los 3 Perfiles de Consumo (Cluster 0: Express, Cluster 1: Variado, Cluster 2: Intensivo)] -->

### 4. Impacto
La adopción de este modelo de segmentación transaccional otorga a Smart Bazar una clara ventaja competitiva y una alta capacidad de respuesta táctica en mostrador y almacén:
* **Diseño Inteligente de Promociones Cruzadas ("Cross-Selling"):** Al caracterizar la canasta multi-ítem del **Cluster 1**, la tienda puede estructurar paquetes promocionales o "Combos Universitarios/Escolares" prepacados (ej. Cuaderno + Resaltador + Lapiceros con 5% de descuento), incrementando artificialmente el `ticket_promedio_item` sin requerir publicidad masiva.
* **Optimización de la Flotilla y Tiempos en Caja ("Queue Management"):** Reconociendo que el **Cluster 0 (Express)** representa un volumen masivo de transacciones de muy bajo monto ($S/. 1.48$), la tienda puede establecer una "Caja Rápida Exclusiva de Fotocopias y Pago Yape con QR" durante las horas pico. Esto evita que el cliente que compra una lista de 10 ítems (Cluster 1) obstruya el flujo veloz de estudiantes que solo necesitan retirar una impresión.
* **Políticas de Fidelización B2B sin DNI:** Para los **39 clientes mayoristas e institucionales** identificados, la gerencia puede habilitar de inmediato un canal de atención preferencial por WhatsApp o cotizaciones con precio mayorista por escala, protegiendo y fidelizando al 4.6% de transacciones que generan el mayor volumen de caja bruta mensual del negocio.
* **Inteligencia sobre la Huella de Consumo Intra-Ticket:** El negocio comprende científicamente la micro-economía de su punto de venta, reemplazando la intuición empírica por un tablero de control no supervisado que monitorea en tiempo real la evolución de sus tres perfiles de consumo dominantes.
