# Sistema de Diseño UI/UX y Paleta de Colores — SmartBazar

Este documento define la guía oficial del **Sistema de Diseño UI/UX** y la **Paleta de Colores** para el Dashboard de Minería de Datos de **SmartBazar**, tomando como referencia la especificación visual y la identidad de marca del proyecto.

---

## 🎨 1. Paleta de Colores (Color Tokens)

La paleta se rige por un esquema cromático sobrio, de alto contraste y sofisticación (Estilo *Monochrome Glassmorphism*):

| Rol de Color | Código Hex / RGBA | Uso e Interpretación en la Interfaz |
| :--- | :--- | :--- |
| **Primary** | `#000000` | Botones de acción principal (Pill CTAs), estado activo del menú lateral, títulos principales (`H1`, `H2`) y elementos de alto impacto. |
| **Secondary** | `#5D5F5F` | Subtítulos, texto secundario, estados inactivosa del menú, bordes finos de tarjetas y descripciones de KPIs. |
| **Tertiary** | `#000000` | Acentos de estructura, bordes de tablas y badges de insights. |
| **Neutral** | `#777777` / `#94a3b8` | Captions, divisores de sección, etiquetas muted y bordes secundarios. |
| **Canvas Background** | `#cbd5e1` → `#f1f5f9` | Fondo general de la aplicación con degradado radial suave y degradados de luz ambiental. |
| **Surface / Glass** | `rgba(255, 255, 255, 0.75)` | Tarjetas contenedoras de gráficos, paneles de control y sidebar con filtro de desfoque `backdrop-filter: blur(20px)`. |
| **Active Text / Pill** | `#ffffff` | Texto blanco puro dentro de botones y pills seleccionados (`#000000`). |

---

## 🔤 2. Sistema Tipográfico (Typography Hierarchy)

Se utiliza la familia tipográfica moderna de sans-serif **Inter** (Google Fonts) para optimizar la legibilidad de datos complejos:

* **Headline (`H1` / `H2`)**:
  * *Fuente*: Inter ExtraBold (Weight 800)
  * *Color*: `#000000`
  * *Letter Spacing*: `-0.03em`
  * *Uso*: Encabezados de paneles, título del Sidebar (**SmartBazar**).

* **Sub-Headline (`H3` / `H4`)**:
  * *Fuente*: Inter Bold (Weight 700)
  * *Color*: `#1e293b`
  * *Uso*: Títulos de tarjetas, pestañas seleccionadas, KPIs numéricos principales.

* **Body Content**:
  * *Fuente*: Inter Regular / Medium (Weight 400 - 500)
  * *Color*: `#5D5F5F` / `#1c1b1b`
  * *Uso*: Párrafos descriptivos, texto explicativo de modelos y tablas de datos.

* **Labels & Badges**:
  * *Fuente*: Inter SemiBold (Weight 600 - 700)
  * *Transformación*: Uppercase (Mayúsculas)
  * *Letter Spacing*: `0.05em`
  * *Uso*: Badges de **INSIGHT DE NEGOCIO**, títulos de KPIs y divisiones de fase del Sidebar.

---

## 🧩 3. Guía de Componentes UI/UX

### A. Barra Lateral de Navegación (Sidebar)
* **Formato de Opción Activa**: Estilo *Pill* totalmente redondeado (`border-radius: 999px`), con fondo negro sólido (`#000000`), texto blanco puro (`#ffffff`) e ícono monocromático SVG integrado.
* **Formato de Opciones Inactivas**: Fondo transparente, texto en escala de grises slate (`#334155`), hover con elevación suave (`#ffffff`) e ícono monocromático en negro.
* **Agrupación**: Organización mediante cabeceras de sección (*Fases del Proyecto*) en fuente `Inter 800` en mayúsculas pequeñas (`#777777`).

### B. Tarjetas de KPIs (Glass Cards)
* **Fondo**: Translúcido `rgba(255, 255, 255, 0.65)` con `backdrop-filter: blur(16px)`.
* **Borde**: `1px solid rgba(255, 255, 255, 0.8)`.
* **Sombra**: `0 4px 20px rgba(0, 0, 0, 0.04)`.
* **Radio de borde**: `14px` - `16px`.

### C. Botones y Controles (Pill Buttons & Tabs)
* **Botón Principal**: Sólido negro `#000000` con `border-radius: 12px` / `999px`, sombra al presionar y transición suave de hover.
* **Tabs (Pestañas)**: Contenedor integrado con borde redondeado `12px`, donde la pestaña seleccionada se resalta en negro con sombra suave.
