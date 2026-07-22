"""Clasificación canónica y reglas de asociación para el Panel 1D."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


PENDING_CATEGORY = "PENDIENTE_REVISION"

# Se evalúan de mayor especificidad a menor especificidad.
KEYWORD_RULES: tuple[tuple[str, str], ...] = (
    ("FOTOCOPIA", "FOTOCOPIA E IMPRESION"),
    ("IMPRESION", "FOTOCOPIA E IMPRESION"),
    ("TIPEO", "SERVICIOS"),
    ("LAPICERO", "LAPICERO"),
    ("PORTAMINAS", "LAPIZ"),
    ("LAPIZ", "LAPIZ"),
    ("CARBONCILLO", "LAPIZ"),
    ("PLUMON", "PLUMON / RESALTADOR"),
    ("RESALTADOR", "PLUMON / RESALTADOR"),
    ("CHEQUEO", "PLUMON / RESALTADOR"),
    ("MARKER", "PLUMON / RESALTADOR"),
    ("INDELEBLE", "PLUMON / RESALTADOR"),
    ("CORRECTOR", "CORRECTOR"),
    ("MINAS", "MINAS / REPUESTOS"),
    ("CRAYON", "CRAYOLAS"),
    ("COLORES", "COLORES"),
    ("TEMPERA", "TEMPERA"),
    ("PINCEL", "PINCEL"),
    ("PALETA", "ACCESORIOS DE PINTURA"),
    ("PLASTILINA", "PLASTILINA"),
    ("LENTEJUELA", "LENTEJUELAS Y ESCARCHA"),
    ("ESCARCHA", "LENTEJUELAS Y ESCARCHA"),
    ("OROPEL", "PAPEL OROPEL"),
    ("MICROPOROSO", "MICROPOROSO"),
    ("BAJA LENGUA", "PALITOS DE MANUALIDADES"),
    ("PALITOS DE CHUPETE", "PALITOS DE MANUALIDADES"),
    ("OJOS MOVILES", "MANUALIDADES"),
    ("PUNZON", "HERRAMIENTAS DE MANUALIDADES"),
    ("ALGODON", "ALGODON"),
    ("SILICONA", "SILICONA"),
    ("GOMA", "GOMA"),
    ("LIMPIATIPO", "LIMPIATIPOS"),
    ("CINTA MASKING", "CINTA ADHESIVA / EMBALAJE"),
    ("CINTA EMBALAJE", "CINTA ADHESIVA / EMBALAJE"),
    ("CINTA DE ESCRITORIO", "CINTA ADHESIVA / EMBALAJE"),
    ("CINTA DE AGUA", "CINTAS DECORATIVAS Y AGUA"),
    ("SKETCH BOOK", "BLOCK / SKETCH BOOK"),
    ("SKETCHBOOK", "BLOCK / SKETCH BOOK"),
    ("BLOCK", "BLOCK / SKETCH BOOK"),
    ("CUADERNO", "CUADERNO"),
    ("POST-IT", "NOTAS ADHESIVAS / POST-IT"),
    ("PAPEL BOND", "HOJA BOND"),
    ("HOJA BOND", "HOJA BOND"),
    ("HOJA DE COLOR", "HOJAS DE COLORES"),
    ("CARTULINA", "CARTULINA"),
    ("PAPEL CREPE", "PAPEL CREPE"),
    ("PAPEL LUSTRE", "PAPEL LUSTRE"),
    ("PAPEL KRAFT", "PAPEL ESPECIAL / KRAFT"),
    ("PAPEL CRAFT", "PAPEL ESPECIAL / KRAFT"),
    ("PAPEL DE SEDA", "PAPEL DE SEDA"),
    ("PAPEL DE SEDE", "PAPEL DE SEDA"),
    ("PAPELOTE", "PAPELOTE"),
    ("SOBRE", "SOBRE"),
    ("FOLDER", "FOLDER"),
    ("MICA", "MICA / FOTOCHECK"),
    ("PLASTIFORRO", "FORRO / VINIFAN"),
    ("VINIFAN", "FORRO / VINIFAN"),
    ("FASTER", "SUJETADORES / FASTERS"),
    ("CHINCHE", "SUJETADORES / CHINCHES"),
    ("GRAPA", "SUJETADORES / GRAPAS"),
    ("LIGAS", "SUJETADORES / LIGAS"),
    ("PABILO", "PABILO"),
    ("TIJERA", "TIJERA / CUTER"),
    ("CUTER", "TIJERA / CUTER"),
    ("TAJADOR", "TAJADOR"),
    ("BORRADOR", "BORRADOR"),
    ("REGLA", "REGLA / GEOMETRIA"),
    ("ESCUADRA", "REGLA / GEOMETRIA"),
    ("TRANSPORTADOR", "REGLA / GEOMETRIA"),
    ("PERFORADOR", "PERFORADOR"),
    ("CALCULADORA", "CALCULADORA"),
    ("MOTA", "MOTA"),
    ("TAMPON", "TAMPON Y TINTAS"),
    ("TINTA", "TAMPON Y TINTAS"),
    ("GLOBO", "GLOBOS Y ACCESORIOS"),
    ("PALIGLOBO", "GLOBOS Y ACCESORIOS"),
    ("SERPENTINA", "SERPENTINA Y PICA PICA"),
    ("PICA PICA", "SERPENTINA Y PICA PICA"),
    ("FLAUTA", "INSTRUMENTOS MUSICALES"),
    ("CORBATA", "ARTICULOS CIVICOS Y UNIFORME"),
    ("ESCARAPELA", "ARTICULOS CIVICOS Y UNIFORME"),
    ("BANDERA", "ARTICULOS CIVICOS Y UNIFORME"),
    ("INSIGNA", "ARTICULOS CIVICOS Y UNIFORME"),
    ("BRIGADIER", "ARTICULOS CIVICOS Y UNIFORME"),
    ("CORREA", "ARTICULOS CIVICOS Y UNIFORME"),
    ("GUANTE", "GUANTES E HIGIENE"),
    ("MANDIL", "MANDIL Y BOLSA"),
    ("BOLSA", "MANDIL Y BOLSA"),
    ("SUPER BLUE", "ACCESORIOS DE LIMPIEZA"),
)

CATEGORY_ALIASES = {
    "LAPICEROS": "LAPICERO", "PLUMONES": "PLUMON / RESALTADOR",
    "RESALTADORES": "PLUMON / RESALTADOR", "INDELEBLE": "PLUMON / RESALTADOR",
    "TEMPERAS": "TEMPERA", "GOMAS": "GOMA", "SILICONAS": "SILICONA",
    "PINCELES": "PINCEL", "REGLAS": "REGLA / GEOMETRIA", "TIJERAS": "TIJERA / CUTER",
    "BORRADORES": "BORRADOR", "TAJADORES": "TAJADOR", "CUADERNOS": "CUADERNO",
    "CARTULINAS": "CARTULINA", "CRAYOLA": "CRAYOLAS", "CRAYOLAS": "CRAYOLAS",
    "CINTAS EMBALAJE": "CINTA ADHESIVA / EMBALAJE", "CINTAS": "CINTA ADHESIVA / EMBALAJE",
    "PAPELERIA": "GENERAL / OTROS", "UTILES": "GENERAL / OTROS", "UNIDAD": "GENERAL / OTROS",
    "COPIA": "FOTOCOPIA E IMPRESION", "COPIAS": "FOTOCOPIA E IMPRESION",
    "IMPRESION": "FOTOCOPIA E IMPRESION", "HOJA DE COLORES": "HOJAS DE COLORES",
    "ESCHARCHAS": "LENTEJUELAS Y ESCARCHA", "ESCARCHAS": "LENTEJUELAS Y ESCARCHA",
}


def normalize_text(value: object) -> str:
    """Devuelve texto comparable, sin tildes, con espacios y mayúsculas canónicos."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip().upper()


def normalize_id(value: object) -> str:
    return re.sub(r"\.0$", "", normalize_text(value))


def classify_product(description: object, inventory_category: object = None) -> tuple[str, str]:
    description_key = normalize_text(description)
    if description_key and description_key != "SIN DESCRIPCION":
        for keyword, category in KEYWORD_RULES:
            if keyword in description_key:
                return category, "DESCRIPCION"

    fallback = CATEGORY_ALIASES.get(normalize_text(inventory_category))
    if fallback:
        return fallback, "INVENTARIO"
    return PENDING_CATEGORY, "PENDIENTE_REVISION"


def audit_inventory_categories(inventory: pd.DataFrame) -> pd.DataFrame:
    audit = inventory[["Categoria"]].copy()
    audit["Categoria_Original"] = audit.pop("Categoria").fillna("<NULO>").astype(str)
    audit["Categoria_Normalizada"] = audit["Categoria_Original"].map(normalize_text)
    audit["Categoria_Canonica_Respaldo"] = audit["Categoria_Normalizada"].map(CATEGORY_ALIASES).fillna(PENDING_CATEGORY)
    audit["Accion"] = "NORMALIZADA"
    invalid = audit["Categoria_Normalizada"].isin({"", "-", "TAILOY"})
    audit.loc[invalid, ["Categoria_Canonica_Respaldo", "Accion"]] = [PENDING_CATEGORY, "REVISAR"]
    return (audit.groupby(["Categoria_Original", "Categoria_Normalizada", "Categoria_Canonica_Respaldo", "Accion"], dropna=False)
            .size().reset_index(name="Productos"))


def prepare_category_data(detail: pd.DataFrame, inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une ventas con inventario y agrega categoría analítica y trazabilidad."""
    inv = inventory[["ID", "Categoria"]].copy()
    inv["_product_id"] = inv["ID"].map(normalize_id)
    inv = inv.drop_duplicates("_product_id")
    sales = detail.copy()
    sales["_product_id"] = sales["ID_Producto"].map(normalize_id)
    enriched = sales.merge(inv[["_product_id", "Categoria"]], on="_product_id", how="left")
    classified = enriched.apply(lambda row: classify_product(row["Descripcion"], row["Categoria"]), axis=1)
    enriched[["Categoria_Analitica", "Fuente_Categoria"]] = pd.DataFrame(classified.tolist(), index=enriched.index)
    return enriched, audit_inventory_categories(inventory)


def category_basket(enriched_sales: pd.DataFrame) -> pd.DataFrame:
    usable = enriched_sales.loc[enriched_sales["Categoria_Analitica"] != PENDING_CATEGORY]
    return pd.crosstab(usable["ID_Venta"], usable["Categoria_Analitica"]).gt(0)


def mine_category_rules(enriched_sales: pd.DataFrame, min_support: float) -> pd.DataFrame:
    baskets = category_basket(enriched_sales)
    columns = ["Antecedente", "Consecuente", "Soporte", "Confianza", "Lift"]
    if baskets.empty or baskets.shape[1] < 2:
        return pd.DataFrame(columns=columns)
    frequent = apriori(baskets, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame(columns=columns)
    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    if rules.empty:
        return pd.DataFrame(columns=columns)
    rules["Antecedente"] = rules["antecedents"].map(lambda items: ", ".join(sorted(items)))
    rules["Consecuente"] = rules["consequents"].map(lambda items: ", ".join(sorted(items)))
    return rules.rename(columns={"support": "Soporte", "confidence": "Confianza", "lift": "Lift"})[columns].sort_values("Lift", ascending=False).reset_index(drop=True)
