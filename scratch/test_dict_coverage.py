import pandas as pd

df = pd.read_csv("datasets/crudo/inventario.csv", sep=";", dtype=str, skiprows=1)
descs = sorted(list(set(df['Descripcion'].dropna().str.strip().str.upper())))
descs = [d for d in descs if d != "" and d != "NAN"]

diccionario_mapeo = {
    # Escritura y Trazado (Ejemplo estricto solicitado en el prompt)
    "LAPIZ 2B": "LAPIZ",
    "PORTAMINAS": "LAPIZ",
    "LÁPIZ": "LAPIZ",
    "LAPIZ": "LAPIZ",
    "CARBONCILLO": "LAPIZ",
    "FRIXION": "LAPICERO",
    "LAPICERO": "LAPICERO",
    "PLUMON": "PLUMON / RESALTADOR",
    "PLUMÓN": "PLUMON / RESALTADOR",
    "RESALTADOR": "PLUMON / RESALTADOR",
    "CHEQUEO": "PLUMON / RESALTADOR",
    "MARKER": "PLUMON / RESALTADOR",
    "INDELEBLE": "PLUMON / RESALTADOR",
    "CORRECTOR": "CORRECTOR",
    "MINAS": "MINAS / REPUESTOS",
    "TIZA": "TIZA",
    
    # Arte, Dibujo y Manualidades
    "LAPICES DE COLOR": "COLORES",
    "COLORES": "COLORES",
    "CRAYON": "CRAYOLAS",
    "CRAYOLA": "CRAYOLAS",
    "TEMPERA": "TEMPERA",
    "TÉMPERA": "TEMPERA",
    "PINCEL": "PINCEL",
    "PALETA": "ACCESORIOS DE PINTURA",
    "PLASTILINA": "PLASTILINA",
    "LENTEJUELA": "LENTEJUELAS Y ESCARCHA",
    "ESCARCHA": "LENTEJUELAS Y ESCARCHA",
    "OROPEL": "PAPEL OROPEL",
    "MICROPOROSO": "MICROPOROSO",
    "PALITOS DE CHUPETE": "PALITOS DE MANUALIDADES",
    "BAJA LENGUA": "PALITOS DE MANUALIDADES",
    "OJOS MOVILES": "MANUALIDADES",
    "PUNZON": "HERRAMIENTAS DE MANUALIDADES",
    "ESPONJA": "ESPONJA",
    "ALGODÓN": "ALGODON",
    "ALGODON": "ALGODON",
    
    # Pegamentos y Adhesivos
    "SILICONA": "SILICONA",
    "GOMA": "GOMA",
    "CINTA MASKING": "CINTA ADHESIVA / EMBALAJE",
    "CINTA EMBALAJE": "CINTA ADHESIVA / EMBALAJE",
    "CINTA DE ESCRITORIO": "CINTA ADHESIVA / EMBALAJE",
    "CINTA DE AGUA": "CINTAS DECORATIVAS Y AGUA",
    "LIMPIATIPO": "LIMPIATIPOS",
    
    # Cuadernos, Blocks y Papelería
    "SKETCH BOOK": "BLOCK / SKETCH BOOK",
    "BLOCK": "BLOCK / SKETCH BOOK",
    "CUADERNO": "CUADERNO",
    "POST-IT": "NOTAS ADHESIVAS / POST-IT",
    "HOJA BOND": "HOJA BOND",
    "PAPEL BOND": "HOJA BOND",
    "HOJA DE COLOR": "HOJAS DE COLORES",
    "HOJAS DE COLOR": "HOJAS DE COLORES",
    "CARTUILINA": "CARTULINA",  # Corrección para error tipográfico del POS
    "CARTULINA": "CARTULINA",
    "PAPEL CREPE": "PAPEL CREPE",
    "PAPEL LUSTRE": "PAPEL LUSTRE",
    "PAPEL KRAFT": "PAPEL ESPECIAL / KRAFT",
    "PAPEL CRAFT": "PAPEL ESPECIAL / KRAFT",
    "PAPEL DE SEDA": "PAPEL DE SEDA",
    "PAPEL DE SEDE": "PAPEL DE SEDA",  # Corrección para error tipográfico del POS
    "PAPEL MANTECA": "PAPEL ESPECIAL",
    "PAPEL DE REGALO": "PAPEL DE REGALO",
    "PAPELOTE": "PAPELOTE",
    "SOBRE": "SOBRE",
    "TARJETA BIBLIOGRAFICA": "TARJETA BIBLIOGRAFICA",
    "BILLETE Y MONEDA": "MATERIAL DIDACTICO",
    "TABLA PERIODICA": "MATERIAL DIDACTICO",
    "LAMINAS": "LAMINAS DIDACTICAS",
    
    # Organización, Archivo y Sujetadores
    "FOLDER": "FOLDER",
    "MICA": "MICA / FOTOCHECK",
    "PLASTIFORRO": "FORRO / VINIFAN",
    "VINIFAN": "FORRO / VINIFAN",
    "FASTER": "SUJETADORES / FASTERS",
    "CHINCHE": "SUJETADORES / CHINCHES",
    "GRAPA": "SUJETADORES / GRAPAS",
    "LIGAS": "SUJETADORES / LIGAS",
    "PABILO": "PABILO",
    
    # Herramientas y Accesorios de Escritorio
    "TIJERA": "TIJERA / CUTER",
    "CUTER": "TIJERA / CUTER",
    "TAJADOR": "TAJADOR",
    "BORRADOR": "BORRADOR",
    "REGLA": "REGLA / GEOMETRIA",
    "ESCUADRA": "REGLA / GEOMETRIA",
    "TRANSPORTADOR": "REGLA / GEOMETRIA",
    "PERFORADOR": "PERFORADOR",
    "CALCULADORA": "CALCULADORA",
    "MOTA": "MOTA",
    "TAMPON": "TAMPON Y TINTAS",
    "TINTA": "TAMPON Y TINTAS",
    "LUPA": "LUPA",
    "PERCHERO": "PERCHERO",
    
    # Artículos Festivos y Recreación
    "GLOBO": "GLOBOS Y ACCESORIOS",
    "PALIGLOBO": "GLOBOS Y ACCESORIOS",
    "SERPENTINA": "SERPENTINA Y PICA PICA",
    "PICA PICA": "SERPENTINA Y PICA PICA",
    "FLAUTA": "INSTRUMENTOS MUSICALES",
    "PITO": "ARTICULOS DEPORTIVOS Y RECREACION",
    
    # Artículos Cívicos y Uniforme
    "CORBATA": "ARTICULOS CIVICOS Y UNIFORME",
    "ESCARAPELA": "ARTICULOS CIVICOS Y UNIFORME",
    "BANDERA": "ARTICULOS CIVICOS Y UNIFORME",
    "INSIGNA": "ARTICULOS CIVICOS Y UNIFORME",
    "PALO DE BRIGADIER": "ARTICULOS CIVICOS Y UNIFORME",
    "CORREA": "ARTICULOS CIVICOS Y UNIFORME",
    "GUANTE": "GUANTES E HIGIENE",
    "MANDIL": "MANDIL Y BOLSA",
    "BOLSA": "MANDIL Y BOLSA",
    
    # Servicios y Otros
    "FOTOCOPIA": "FOTOCOPIA E IMPRESION",
    "IMPRESION": "FOTOCOPIA E IMPRESION",
    "TIPEO": "SERVICIOS",
    "SUPER BLUE": "ACCESORIOS DE LIMPIEZA",
    "PRODUCTO": "GENERAL / OTROS"
}

def categorizar_producto(descripcion, diccionario=diccionario_mapeo):
    if not isinstance(descripcion, str):
        return "OTRO"
    desc_upper = descripcion.upper()
    for clave, categoria in diccionario.items():
        if clave in desc_upper:
            return categoria
    return "OTRO"

sin_mapeo = []
mapeados = {}
for d in descs:
    cat = categorizar_producto(d)
    mapeados[d] = cat
    if cat == "OTRO":
        sin_mapeo.append(d)

print(f"Total productos evaluados: {len(descs)}")
print(f"Productos sin mapear (categoría 'OTRO'): {len(sin_mapeo)}")
if sin_mapeo:
    print("Muestra de no mapeados:")
    for sm in sin_mapeo[:20]:
        print(" -", sm)
