# Reglas de Asociación: lógica, algoritmos y trazabilidad

## 1. Propósito de la vista

La vista **Reglas de Asociación** realiza *Market Basket Analysis*: busca categorías de productos que aparecen juntas en los mismos tickets y expresa esas relaciones como reglas de la forma:

$$A \rightarrow B$$

La regla no significa causalidad. Significa que, dentro del historial analizado, la presencia de las categorías del antecedente $A$ está estadísticamente asociada con la presencia de las categorías del consecuente $B$.

La unidad de análisis es el **ticket** (`ID_Venta`), no el cliente, la unidad vendida ni el importe. Por eso esta vista responde preguntas como «¿en qué proporción de tickets aparecen juntas CARTULINA y SILICONA?», pero no «¿cuántas unidades se compraron?» ni «¿cuánto dinero generó el combo?».

### Etiquetas de procedencia usadas en este documento

| Etiqueta | Significado |
|---|---|
| **En vivo** | Se calcula al ejecutar o reejecutar la vista con los CSV y controles disponibles en ese momento. |
| **Precalculado** | Fue calculado fuera de Streamlit y se lee como un resultado ya producido. |
| **Preentrenado** | Proviene de un modelo entrenado y persistido con anterioridad. |
| **Hardcodeado** | Es una regla, catálogo, texto o cifra escrita directamente en el código. |

En esta vista no existe un modelo preentrenado ni un archivo de resultados: Apriori se ejecuta en vivo. Sin embargo, la taxonomía usada para categorizar productos sí está hardcodeada.

## 2. Ruta activa y archivos involucrados

```text
app.py
  opción "Reglas de Asociación"
    -> render_association_panel()
       -> load_association_data()
          -> load_detalle_ventas() / load_inventario()
          -> prepare_category_data()
       -> mine_category_rules()
          -> category_basket()
          -> mlxtend.apriori()
          -> mlxtend.association_rules()
       -> KPIs, dispersión, grafo, tabla y auditoría
```

| Archivo | Funciones o elementos | Responsabilidad real |
|---|---|---|
| `app.py`, líneas 10-15 y 345-460 | `load_association_data()`, `render_association_panel()` | Orquesta carga, controles, minería, filtrado y presentación de todos los resultados activos. |
| `app.py`, líneas 478-501 | menú y enrutamiento | La opción llama directamente a `render_association_panel()`. |
| `src/data_loader.py`, líneas 5-22 y 138-153 | `get_datasets_path()`, `_find_file()`, `load_detalle_ventas()`, `load_inventario()` | Resuelve y lee `datasets/limpio/detalle_ventas.csv` e `inventario.csv`; usa los crudos solo si el limpio no existe. |
| `src/category_rules.py`, líneas 12-119 | `PENDING_CATEGORY`, `KEYWORD_RULES`, `CATEGORY_ALIASES` | Define la taxonomía y sus respaldos de forma hardcodeada. |
| `src/category_rules.py`, líneas 122-170 | normalización, clasificación, auditoría y unión | Convierte las líneas de venta a categorías analíticas con trazabilidad. |
| `src/category_rules.py`, líneas 173-191 | `category_basket()`, `mine_category_rules()` | Construye la matriz binaria, ejecuta Apriori y genera las reglas. |
| `Notebooks/Panel 1D_Reglas_Asociacion.ipynb` | análisis offline | Explora SKU, categoría y departamento con Apriori y FP-Growth. No alimenta la vista mediante un artefacto. |
| `src/panel_asociacion.py` | `_build_rules_df()`, `show_panel()` | Implementación antigua con ocho reglas simuladas. Se importa como `show_asociacion`, pero nunca se llama desde el enrutamiento activo. |

La versión actual de los CSV contiene 939 cabeceras de venta, 1,240 líneas de detalle y 417 filas de inventario. Son cifras del snapshot del repositorio, no constantes del algoritmo.

## 3. Preparación y clasificación de categorías

### 3.1 Carga y caché

`load_association_data()` está decorada con `@st.cache_data`. En la primera ejecución de la sesión carga los dos CSV y ejecuta `prepare_category_data()`; en reejecuciones compatibles reutiliza el resultado cacheado. Los sliders vuelven a minar y filtrar reglas, pero no necesariamente fuerzan una nueva lectura del disco. Si un CSV cambia mientras el servidor continúa activo, puede ser necesario limpiar la caché o reiniciar Streamlit.

Los loaders de archivos limpios usan `pd.read_csv(..., encoding="utf-8")` sin repetir el saneamiento de los datos crudos. Por tanto, la vista confía en que el proceso anterior ya produjo correctamente `datasets/limpio/`.

### 3.2 Normalización

`normalize_text(value)` aplica tres operaciones:

1. Los nulos se transforman en cadena vacía.
2. Unicode NFKD separa letras y tildes; luego se eliminan las marcas diacríticas. Por ejemplo, `LÁPIZ` pasa a `LAPIZ`.
3. Se colapsan espacios repetidos, se recortan extremos y se convierte a mayúsculas.

`normalize_id(value)` reutiliza esa normalización y elimina un sufijo `.0`. Así, un ID leído como `7750082416178.0` puede unirse con `7750082416178`.

### 3.3 Unión detalle-inventario

`prepare_category_data(detail, inventory)` sigue este flujo:

1. Conserva `ID` y `Categoria` del inventario.
2. Crea `_product_id` normalizado.
3. Ejecuta `drop_duplicates("_product_id")`; si el inventario contiene varias filas con el mismo ID, gana silenciosamente la primera según el orden del archivo.
4. Normaliza `ID_Producto` del detalle.
5. Realiza un `left merge`, por lo que ninguna línea de venta se elimina aunque no encuentre inventario.
6. Aplica `classify_product()` fila por fila y agrega `Categoria_Analitica` y `Fuente_Categoria`.

### 3.4 Precedencia de clasificación

`classify_product(description, inventory_category)` aplica esta prioridad:

1. **Descripción — Hardcodeado + En vivo.** Recorre `KEYWORD_RULES` en orden y devuelve la categoría del primer texto contenido en la descripción normalizada. El orden es parte del algoritmo: se declaran reglas más específicas antes que reglas generales.
2. **Categoría de inventario — Hardcodeado + En vivo.** Si ninguna palabra clave coincide, normaliza la categoría del inventario y busca una equivalencia exacta en `CATEGORY_ALIASES`.
3. **Pendiente.** Si tampoco existe alias, asigna `PENDIENTE_REVISION`.

Ejemplo: una descripción con `FOTOCOPIA` se clasifica como `FOTOCOPIA E IMPRESION` aunque el inventario diga otra cosa. `Fuente_Categoria` permite saber si el resultado vino de `DESCRIPCION`, `INVENTARIO` o `PENDIENTE_REVISION`.

La coincidencia es por **subcadena**, no por token completo ni por un modelo NLP. Esto permite tolerar descripciones largas, pero puede producir falsos positivos si una palabra clave aparece dentro de otro término. Incorporar una nueva categoría requiere editar el código y respetar el orden de precedencia.

### 3.5 Auditoría del catálogo

`audit_inventory_categories()` no audita el resultado final de cada línea vendida; resume únicamente el campo `Categoria` de inventario:

- `Categoria_Original`: texto original o `<NULO>`.
- `Categoria_Normalizada`: versión sin tildes, espacios sobrantes y en mayúsculas.
- `Categoria_Canonica_Respaldo`: resultado de `CATEGORY_ALIASES` o `PENDIENTE_REVISION`.
- `Accion`: `REVISAR` solo si la categoría normalizada es `""`, `"-"` o `"TAILOY"`; en los demás casos queda `NORMALIZADA`.
- `Productos`: número de filas de inventario en cada combinación.

Una limitación importante es que una categoría desconocida —e incluso `<NULO>` después de `fillna()`— puede quedar con acción `NORMALIZADA` y respaldo `PENDIENTE_REVISION`. Por ello se deben leer juntas las columnas `Accion` y `Categoria_Canonica_Respaldo`.

## 4. Construcción de la canasta binaria

`category_basket()` elimina primero las líneas cuya categoría sea `PENDIENTE_REVISION` y después ejecuta:

```python
pd.crosstab(usable["ID_Venta"], usable["Categoria_Analitica"]).gt(0)
```

El resultado es una matriz booleana $X$:

$$x_{tc}=\begin{cases}
1,&\text{si el ticket }t\text{ contiene al menos una línea de la categoría }c\\
0,&\text{en caso contrario}
\end{cases}$$

Consecuencias metodológicas:

- Dos o veinte unidades de una categoría producen el mismo `True`.
- Varias líneas de la misma categoría en un ticket cuentan una sola vez.
- `Cantidad`, `Precio_Unitario`, `Subtotal`, `Fecha`, `Medio` y el orden de compra no intervienen.
- Una línea pendiente se omite, pero las otras categorías utilizables del ticket permanecen.
- Un ticket compuesto únicamente por líneas pendientes desaparece por completo de la matriz.

Por tanto, el denominador $N$ del soporte es el número de tickets con **al menos una categoría utilizable**, no necesariamente las 939 cabeceras ni todos los `ID_Venta` del detalle.

## 5. Algoritmo Apriori

### 5.1 Itemsets frecuentes

`mine_category_rules()` llama a `mlxtend.frequent_patterns.apriori()` con la matriz binaria, el soporte elegido y `use_colnames=True`.

Para un conjunto de categorías $X$:

$$soporte(X)=\frac{|\{t:X\subseteq t\}|}{N}$$

Apriori usa la propiedad antimonótona: si un conjunto no alcanza el soporte mínimo, ningún superconjunto que lo contenga podrá alcanzarlo. El algoritmo genera candidatos por niveles (categorías individuales, pares, ternas, etc.) y poda ramas infrecuentes. Esto evita evaluar todas las combinaciones posibles.

El slider `Soporte Mínimo` se pasa directamente a Apriori. Bajar el umbral normalmente produce más itemsets y más reglas, aumenta el costo de cómputo y puede resaltar coincidencias raras; subirlo conserva patrones más extendidos pero puede ocultar nichos útiles.

### 5.2 Reglas y métricas

Si existen itemsets frecuentes, `association_rules(..., metric="lift", min_threshold=1.0)` genera reglas con lift mayor o igual que 1. La vista no genera asociaciones negativas.

Para una regla $A\rightarrow B$, con $A$ y $B$ disjuntos:

$$soporte(A\rightarrow B)=P(A\cap B)$$

$$confianza(A\rightarrow B)=P(B|A)=\frac{P(A\cap B)}{P(A)}$$

$$lift(A\rightarrow B)=\frac{P(B|A)}{P(B)}=\frac{P(A\cap B)}{P(A)P(B)}$$

Interpretación:

- **Soporte:** representatividad del combo en todos los tickets utilizables.
- **Confianza:** entre los tickets con $A$, proporción que también contiene $B$.
- **Lift > 1:** $B$ aparece con $A$ más de lo esperado bajo independencia.
- **Lift = 1:** comportamiento compatible con independencia.
- **Lift < 1:** asociación negativa; no puede aparecer en esta vista por el umbral interno.

Una confianza alta no basta: si $B$ es muy frecuente, puede aparecer casi siempre con cualquier antecedente. El lift corrige esa frecuencia base. A su vez, un lift muy alto con soporte mínimo puede representar pocos tickets, de modo que las tres métricas deben interpretarse conjuntamente.

Los conjuntos `frozenset` se convierten a texto ordenado y separado por comas. Una regla puede contener varias categorías en ambos lados. El resultado final conserva solo `Antecedente`, `Consecuente`, `Soporte`, `Confianza` y `Lift`, y queda ordenado por lift descendente.

### 5.3 Salidas vacías

La función devuelve una tabla vacía con el esquema correcto cuando:

- no existen tickets utilizables;
- hay menos de dos categorías;
- Apriori no encuentra itemsets con el soporte solicitado; o
- no se genera ninguna regla con lift mínimo 1.

Esto permite que la interfaz muestre mensajes en el grafo y la tabla. La pestaña de dispersión, en cambio, muestra ejes vacíos sin un mensaje específico.

## 6. Controles y orden de filtrado

| Control | Rango / defecto | Momento de aplicación | Efecto |
|---|---:|---|---|
| `Soporte Mínimo` | 0.005–0.10; defecto 0.005; paso 0.005 | Dentro de Apriori | Cambia qué itemsets y reglas llegan a existir. |
| `Confianza Mínima` | 0.05–1.00; defecto 0.40; paso 0.05 | Después de generar reglas | Filtra el DataFrame; no reduce la búsqueda de itemsets. |
| `Lift Mínimo` | 1.0–10.0; defecto 1.0; paso 0.1 | Después de generar reglas | Filtra reglas ya restringidas internamente a lift ≥ 1. |

Cada cambio provoca una reejecución de Streamlit. Primero se minera con soporte, después se exige simultáneamente `Confianza >= min_conf` y `Lift >= min_lift`. Todos los KPIs y gráficos, salvo cobertura y auditoría, usan este conjunto filtrado.

## 7. Inventario de resultados de la vista

| Resultado visible | Procedencia | Cálculo exacto | Cómo interpretarlo / advertencias |
|---|---|---|---|
| **Reglas Activas** | **En vivo** | `len(reglas_filt)` después de los tres umbrales. | Cuenta direcciones, no pares únicos: $A\to B$ y $B\to A$ son dos reglas. El subtítulo solo menciona lift aunque soporte y confianza también influyen. |
| **Mayor Lift** | **En vivo** | Primera fila filtrada, porque `mine_category_rules()` ordena por lift descendente. | Muestra el lift y solo los primeros 22 caracteres del antecedente; omite el consecuente. Si no hay reglas muestra `—`. |
| **Cobertura** | **En vivo** | $1-\frac{\text{líneas pendientes}}{\text{líneas enriquecidas}}$. | Es cobertura de **líneas**, no de tickets ni de SKUs. Se marca en rojo si hay al menos una pendiente. Con cero líneas se produciría división por cero. |
| **Dispersión confianza vs soporte** | **En vivo** | Un punto por regla filtrada; $x=$ soporte, $y=$ confianza, color y tamaño $=lift$ (`70 × lift`). | Facilita comparar alcance, conversión y fuerza. Etiqueta solo las primeras seis reglas por lift y trunca ambos nombres a 12 caracteres. |
| **Barra de color Lift** | **En vivo** | Escala `Greys` de los lifts presentes. | El tono es relativo al conjunto filtrado actual, no una escala fija entre ejecuciones. |
| **Grafo de Red** | **En vivo** | `DiGraph` con las primeras 15 reglas por lift; arista antecedente → consecuente y grosor igual al lift. `spring_layout(seed=42)`. | La dirección expresa la regla, no causalidad. En reglas multielemento, todo el texto `A, B` es un nodo compuesto; no se desagrega en categorías individuales. |
| **Tabla de Reglas** | **En vivo** | Todas las reglas filtradas, ordenadas por lift; soporte/confianza como porcentaje y lift con dos decimales. | Es la salida más fiel para lectura exacta. No muestra conteos absolutos, antecedent support, leverage, conviction ni número de tickets. |
| **Auditoría de Categorías** | **En vivo sobre reglas hardcodeadas** | `audit_inventory_categories()` agrupada y ordenada por `Accion` ascendente y `Productos` descendente. | Describe posibles respaldos del catálogo, no demuestra qué fuente clasificó cada línea vendida. |
| **Productos que requieren revisión manual** | **En vivo** | Combinaciones distintas de `ID_Producto`, `Descripcion` y `Categoria` para líneas finales pendientes. | Solo aparece si hay pendientes. Es la lista operativa para ampliar palabras clave o alias. |

En `render_association_panel()` también se calcula `top_conf`, la regla de mayor confianza, pero la variable no se usa en ningún componente visible.

## 8. Qué pertenece al notebook y qué pertenece a Streamlit

`Notebooks/Panel 1D_Reglas_Asociacion.ipynb` es un laboratorio offline más amplio:

- construye matrices en tres granularidades: SKU exacto, categoría universal y departamento;
- filtra SKU raros, estudia sensibilidad al soporte y calcula sparsidad;
- compara **Apriori** con **FP-Growth** y mide tiempo de ejecución;
- produce heatmaps, redes, barras, radar y reglas accionables;
- actualmente importa `prepare_category_data()` para compartir la categorización canónica.

La vista activa no abre el notebook ni lee sus salidas, no ejecuta FP-Growth y no muestra análisis por SKU o departamento. El notebook es evidencia metodológica y un entorno de experimentación; la fuente de verdad del panel es `app.py` + `src/category_rules.py` + los CSV limpios.

`src/panel_asociacion.py` contiene ocho reglas completamente hardcodeadas, métricas de leverage/convicción y una pestaña de combos. Aunque `app.py` importa su `show_panel` con el alias `show_asociacion`, el bloque de enrutamiento llama a `render_association_panel()`. Por ello, sus reglas, controles y gráficos mock **no influyen** en lo que ve el usuario.

## 9. Limitaciones, supuestos y riesgos de interpretación

1. **Asociación no es causalidad.** Una regla puede deberse a temporada escolar, ubicación de productos o un tercer artículo no observado.
2. **Sin secuencia temporal.** El ticket es un conjunto; no se sabe qué producto motivó la compra ni cuál fue añadido después.
3. **Sin intensidad económica.** Cantidades e importes se descartan. Un ticket de una unidad pesa igual que uno mayorista.
4. **Taxonomía manual.** Las palabras clave y alias no se aprenden de datos y requieren mantenimiento. Primera coincidencia gana.
5. **Sesgo por pendientes.** Categorías pendientes se omiten; tickets enteramente pendientes salen del denominador. Esto puede modificar soportes.
6. **Lift mínimo estructural.** No se pueden estudiar sustitución o incompatibilidad porque las reglas con lift menor que 1 se descartan antes del slider.
7. **Rareza estadística.** Con soporte 0.005, una regla puede descansar en muy pocos tickets. No se muestran intervalos de confianza ni pruebas de estabilidad.
8. **Sin validación temporal.** Las reglas se calculan sobre todo el archivo y no se comprueba si se repiten en otro periodo.
9. **Caché.** Los CSV modificados durante la sesión podrían no reflejarse inmediatamente.
10. **Grafo simplificado.** Limita a 15 reglas, usa grosor crudo de lift y fusiona nodos con la misma etiqueta; no es un inventario completo.

## 10. Guion académico para explicar la construcción

1. «Primero convierto descripciones heterogéneas a categorías comparables mediante normalización y reglas con precedencia; conservo la fuente para auditoría».
2. «Después transformo el detalle a una matriz ticket × categoría, donde cada celda indica presencia o ausencia, no cantidad».
3. «Apriori descarta combinaciones infrecuentes usando la propiedad antimonótona y conserva los itemsets que superan el soporte seleccionado».
4. «De esos itemsets genero reglas dirigidas y evalúo alcance con soporte, conversión con confianza y asociación relativa con lift».
5. «Los sliders separan dos fases: soporte cambia la minería; confianza y lift filtran las reglas ya generadas».
6. «Finalmente interpreto reglas con las tres métricas y reviso cobertura, porque una taxonomía incompleta puede sesgar los resultados».

La conclusión correcta no es «comprar A causa comprar B», sino «en este snapshot de tickets utilizables, A y B coocurren con una frecuencia relativa que supera la esperada bajo independencia; el patrón debe validarse antes de convertirlo en promoción».
