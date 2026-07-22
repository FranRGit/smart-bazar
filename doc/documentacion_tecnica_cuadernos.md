# Documentación Técnica de Cuadernos: Proyecto Smart Bazar
**Área:** Minería de Datos & Inteligencia de Negocios (UNMSM 2026-I)  
**Objetivo del Documento:** Guía técnica clara y accesible para el equipo de trabajo explicativa de la metodología, hallazgos y valor de negocio de los Paneles 1A, 1B y 1C.

---

## Panel 1A: Auditoría y Limpieza de Datos

### ¿Qué se ha hecho?
Se realizó una inspección estructural profunda sobre los tres conjuntos de datos crudos del sistema POS de Smart Bazar (`ventas.csv`, `detalle-ventas.csv` e `Inventario.csv`) ubicados en `datasets/crudo/`:
* **Saneamiento estructural de archivos:** Se identificaron y corrigieron problemas típicos de exportación del punto de venta, como el prefijo BOM (`\ufeff`), filas iniciales con delimitadores fantasma y columnas vacías residuales (`Unnamed: 0`), cargando el inventario con `skiprows=1` y aplicando expresiones regulares de limpieza.
* **Eliminación de variables redundantes / Fuga de datos:** En la tabla `detalle-ventas.csv`, se detectó y eliminó la columna `Medio`. Tras auditar la relación relacional, se comprobó que el método de pago formal de cada transacción ya residía de manera completa en la cabecera (`ventas.csv`). Retirar `Medio` del detalle eliminó redundancias contables y previno riesgos de multicolinealidad o duplicidad en fases predictivas.
* **Tipificación y saneamiento numérico:** En `Inventario.csv`, los campos monetarios (`Costo_Unitario`, `Precio_Venta`) y de conteo (`Stock_Actual`, `Stock_Minimo`) venían formateados como cadenas de texto con símbolos monetarios (`S/.`) y comas de miles. Se desarrollaron rutinas de conversión canónica a tipo flotante (`float64`) y se resolvió la inconsistencia de registros en blanco.
* **Sincronización referencial:** Se validó la integridad transaccional uniendo `ID` (cabecera de venta) con `ID_Venta` (detalle), garantizando que no existan huérfanos transaccionales en el flujo de datos.

### ¿Qué se obtuvo?
* **Dataset de Ventas (`ventas.csv`):** Auditoría sobre 854 registros totales que arrojó **841 transacciones plenamente válidas y verificadas**, con 13 registros nulos o mal digitados identificados y corregidos (alcanzando un **98.5% de cobertura real** del historial).
* **Dataset de Detalle de Ventas (`detalle-ventas.csv`):** 1,146 registros limpios y validados, con 0 columnas fantasma y estructura relacional perfecta con respecto a las ventas principales.
* **Dataset de Inventario (`Inventario.csv`):** 929 registros válidos, consolidando un catálogo limpio con más de 900 productos distribuidos formalmente en sus respectivos departamentos (Papelería, Útiles Escolares, Fotocopias, etc.).

### Impacto en el Negocio
* **Confiabilidad operativa y contable:** Elimina la "basura digital" generada por el software POS actual, permitiendo que el dueño y el administrador observen cifras reales de dinero y unidades en inventario sin distorsiones por errores tipográficos o duplicidades.
* **Base sólida para ciencia de datos:** Sienta los cimientos limpios en `datasets/limpio/` sobre los que trabajan en paralelo los modelos predictivos de caja (Panel 2: Yape vs Efectivo) y pronósticos de ingresos diarios (Panel 3: Prophet), evitando fallos en cascada por datos corruptos.

---

## Panel 1B: EDA e Ingeniería de Características

### ¿Qué se ha hecho?
Se desarrolló un Exploratorio Estadístico (EDA) multi-nivel y se sintetizaron nuevas variables transaccionales de alto valor analítico:
* **Inspección Visual Dual (Univariada y Bivariada):** Se evalúan las distribuciones de monto y cantidad de ítems mediante diagramas de caja y densidad (`Boxplot + KDE`), junto con pruebas de hipótesis bivariadas mediante regresión (`regplot`) para entender la relación entre variedad de canasta e ingresos monetarios (`Subtotal`).
* **Auditoría de Tres Fenómenos Operacionales del Negocio:**
  1. *Fenómeno 1 (Auditoría Temporal y Sesgo Horario):* Analizamos los ingresos de mercadería por días de la semana y horas del día para detectar el sesgo humano en el registro (digiting bias).
  2. *Fenómeno 2 (Desfase Contable en Kardex y Stocks Negativos):* Rastreamos anomalías críticas en el catálogo donde el stock actual es menor a cero (`Stock_Actual < 0`), identificando los **8 SKUs más problemáticos** (ej. "Resaltador Pelikan" con -6, "Súper nota Posa" con -5, etc.).
  3. *Fenómeno 3 (Omisión del 76.4% en Stock_Minimo):* Se diagnosticó que el 76.4% de los productos en tienda carecen de un umbral mínimo de reposición configurado en el POS. Se crearon e implementaron **reglas de negocio dinámicas por rotación de categoría y cuartiles** para imputar automáticamente estos umbrales mínimos.
* **Ingeniería de Características por Ticket:** Al agrupar el detalle de ventas por cada `ID_Venta`, se construyeron 8 variables de síntesis transaccional: `total_monto`, `n_items`, `diversidad_productos` (cantidad de SKUs distintos), `ticket_promedio_item` (`total_monto / n_items`), `ratio_diversidad` (`diversidad_productos / n_items`), `n_departamentos`, `max_subtotal` y `std_subtotal`.
* **Análisis de Multicolinealidad (Matriz de Spearman):** Se calculó la correlación no paramétrica de Spearman para las 8 variables sintéticas, identificando fuertes acoplamientos no lineales (como la correlación $\rho = 0.81$ entre `total_monto` y `n_items`) que dictan qué variables usar en modelos posteriores sin caer en redundancia.

### ¿Qué se obtuvo?
* **Aislamiento de fugas y quiebres logísticos:** Detección exacta de los 8 productos con stock negativo producto de descargas sin ingreso previo en Kardex, y subsanación completa del 76.4% de vacíos en políticas de reposición de inventario.
* **Matriz transaccional enriquecida:** Un conjunto de datos a nivel de ticket ($N=841$) con 8 características que capturan no solo cuánto gasta el cliente, sino **cómo compone su carrito de compra**.
* **Mapa de correlación no paramétrica:** La confirmación estadística de que en Smart Bazar el incremento del ticket de venta está impulsado tanto por compras de volumen intensivo como por variedad de canasta escolar, permitiendo discriminar variables para clustering.

### Impacto en el Negocio
* **Prevención de quiebres de stock y pérdidas:** La corrección del `Stock_Minimo` dota a Smart Bazar de una alarma preventiva real. El negocio ya no esperará a quedarse sin papel bond o lapiceros en plena temporada escolar para realizar pedidos a proveedores.
* **Inteligencia sobre el comportamiento de compra:** Permite entender al cliente de mostrador a través de su ticket promedio por ítem y su diversidad, revelando si la tienda funciona en ese instante como un centro de fotocopiado rápido o como una librería de abastecimiento escolar general.

---

## Panel 1C: Clustering (Segmentación de Clientes por Ticket)

### ¿Qué se ha hecho?
Debido a que el POS de Smart Bazar registra el campo `ID_Cliente` de forma general o anónima en la gran mayoría de transacciones al mostrador, se diseñó un **clustering no supervisado basado en el perfil del ticket** utilizando las variables de ingeniería del Panel 1B:
* **Saneamiento de Outliers (Corte de Tukey $1.5 \cdot \text{IQR}$):** Se aplicó el criterio de Tukey sobre `total_monto` y `n_items` para separar la transaccionalidad normal ($N=802$) de aquellas compras excepcionales o de volumen corporativo/mayorista ($N=39$).
* **Estandarización Z-Score (`StandardScaler`):** Para evitar que el monto en Soles domine artificialmente sobre el conteo de ítems o ratios de diversidad, se normalizaron las 8 variables numéricas a media 0 y desviación estándar 1.
* **Evaluación Comparativa de Algoritmos:** Se entrenaron, contrastaron y validaron tres familias de modelos de aprendizaje no supervisado:
  1. **K-Means** (probando $K \in [2, 10]$ con Método del Codo y Coeficiente de Silueta).
  2. **DBSCAN** (ensayando radios $\epsilon$ y `min_samples` basados en distancias vecinas).
  3. **Gaussian Mixture Models (GMM)** (probando agrupamientos probabilísticos con matrices de covarianza completas).

### ¿Qué se obtuvo?
* **Selección Contundente de K-Means ($K=3$):** El modelo K-Means con 3 clusters alcanzó un **Coeficiente de Silueta óptimo de 0.5187**, superando drásticamente el desempeño de DBSCAN (Silueta: `0.2543`, propenso a colapsar en un solo clúster por densidad y clasificar el resto como ruido) y de GMM (Silueta: `0.4812`, con elipsoides traslapados).
* **Identificación de 39 Compras Mayoristas/Institucionales:** Aislamiento exitoso del 4.6% de transacciones que representan pedidos especiales de alto monto (hasta S/. 133) y alto volumen (hasta 24 ítems en un solo ticket).
* **Definición de 3 Perfiles de Consumo Minorista (802 tickets normales):**
  * **Cluster 0 (Compra Express / Ticket Bajo):** Ticket promedio de **S/. 1.48**, dominado por 1 ítem de rápida atención (ej. impresiones rápidas, fotocopias individuales o un lapicero suelto).
  * **Cluster 1 (Canasta Variada / Escolar):** Ticket promedio de **S/. 5.56**, con un promedio de 4 ítems distintos por ticket y alto ratio de diversidad (clientes que compran lista de útiles combinada o material de oficina variado).
  * **Cluster 2 (Compra Intensiva / Repetición de Volumen):** Ticket promedio de **S/. 3.20**, caracterizado por baja diversidad pero múltiples unidades de un mismo SKU (ej. compra de resmas o paquetes de cuadernillos por parte de un docente o estudiante universitario).

### Impacto en el Negocio
* **Estrategia Comercial Sin Necesidad de Registro de Clientes:** Smart Bazar obtiene una radiografía precisa de su clientela sin obligar a los compradores a identificarse con DNI o registrarse, agilizando la cola en el mostrador.
* **Optimización Táctica de Ventas y Promociones:**
  * Para el **Cluster 0 (Express)**, se justifica colocar exhibidores de compra por impulso (golosinas, notas adhesivas, resaltadores) junto a la fotocopiadora.
  * Para el **Cluster 1 (Variado)**, se pueden diseñar promociones tipo "Pack Escolar" o descuentos por llevar artículos complementarios de diferentes departamentos.
  * Para los **39 Mayoristas Detectados**, la gerencia puede implementar una línea de atención directa y políticas de fidelización institucional o cupones de descuento por volumen recurrente.
