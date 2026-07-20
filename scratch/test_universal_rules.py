import pandas as pd
import numpy as np
import os
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

def generalizar_producto(desc):
    if not isinstance(desc, str):
        return "OTRO"
    d = desc.upper().strip()
    
    if "SILICONA" in d:
        return "SILICONA"
    if "FOTOCOPIA" in d:
        return "FOTOCOPIA"
    if "IMPRESION" in d:
        return "IMPRESION"
    if "CUADERNO" in d:
        return "CUADERNO"
    if "LAPICERO" in d:
        return "LAPICERO"
    if "LAPIZ" in d or "LÁPIZ" in d or "PORTAMINAS" in d:
        return "LAPIZ"
    if "GOMA" in d:
        return "GOMA"
    if "BORRADOR" in d:
        return "BORRADOR"
    if "PLUMON" in d or "PLUMÓN" in d or "INDELEBLE" in d or "RESALTADOR" in d or "CHEQUEO" in d or "MARKER" in d:
        return "PLUMON / RESALTADOR"
    if "CARTULINA" in d:
        return "CARTULINA"
    if "MICROPOROSO" in d:
        return "MICROPOROSO"
    if "HOJA DE COLOR" in d or "HOJAS DE COLORES" in d:
        return "HOJAS DE COLORES"
    if "HOJA BOND" in d:
        return "HOJA BOND"
    if "CINTA" in d:
        return "CINTA ADHESIVA / EMBALAJE"
    if "PINCEL" in d:
        return "PINCEL"
    if "MICA" in d:
        return "MICA"
    if "PAPEL CREPE" in d:
        return "PAPEL CREPE"
    if "PAPEL LUSTRE" in d:
        return "PAPEL LUSTRE"
    if "PAPEL DE SEDA" in d or "PAPEL CRAFT" in d or "PAPEL MANTECA" in d:
        return "PAPEL ESPECIAL"
    if "PAPELOTE" in d:
        return "PAPELOTE"
    if "REGLA" in d or "TRANSPORTADOR" in d:
        return "REGLA / GEOMETRIA"
    if "TEMPERA" in d or "TÉMPERA" in d:
        return "TEMPERA"
    if "COLORES" in d:
        return "COLORES"
    if "CRAYOLA" in d:
        return "CRAYOLA"
    if "CORBATA" in d:
        return "CORBATA"
    if "ESCARAPELA" in d:
        return "ESCARAPELA"
    if "GUANTE" in d:
        return "GUANTE"
    if "BANDERA" in d:
        return "BANDERA"
    if "BLOCK" in d or "SKETCH BOOK" in d:
        return "BLOCK / SKETCH BOOK"
    if "FOLDER" in d:
        return "FOLDER"
    if "GLOBO" in d or "PALIGLOBO" in d:
        return "GLOBOS Y ACCESORIOS"
    if "TAJADOR" in d:
        return "TAJADOR"
    if "SOBRE" in d:
        return "SOBRE"
    if "LENTEJUELA" in d:
        return "LENTEJUELAS"
    if "CORREA" in d:
        return "CORREA"
    if "LIMPIATIPO" in d:
        return "LIMPIATIPOS"
    if "PABILO" in d:
        return "PABILO"
    if "MOTA" in d:
        return "MOTA"
    if "LUPA" in d:
        return "LUPA"
    if "TIJERA" in d or "CUTER" in d:
        return "TIJERA / CUTER"
    if "PLASTILINA" in d:
        return "PLASTILINA"
    if "CHINCHE" in d or "GRAPA" in d:
        return "SUJETADORES / CHINCHES"
    if "VINIFAN" in d:
        return "FORRO / MICA VINIFAN"
    if "SUPER" in d:
        return "SUPER BLUE"
    
    palabras = d.split()
    if len(palabras) > 0:
        return palabras[0]
    return "OTRO"

DIR_LIMPIO = 'datasets/limpio/'
df_detalle = pd.read_csv(os.path.join(DIR_LIMPIO, 'detalle_ventas.csv'))
df_detalle['Producto_Universal'] = df_detalle['Descripcion'].apply(generalizar_producto)

transacciones_univ = df_detalle.groupby('ID_Venta')['Producto_Universal'].apply(lambda x: list(set(x))).tolist()

te_univ = TransactionEncoder()
te_univ_ary = te_univ.fit(transacciones_univ).transform(transacciones_univ)
df_trans_univ = pd.DataFrame(te_univ_ary, columns=te_univ.columns_)

soporte_univ = 0.005

frecuentes_univ_ap = apriori(df_trans_univ, min_support=soporte_univ, use_colnames=True)
reglas_univ_ap = association_rules(frecuentes_univ_ap, metric="lift", min_threshold=1.0)

with open("scratch/univ_rules_out5.txt", "w", encoding="utf-8") as out:
    out.write(f"Total Itemsets Frecuentes (min_support={soporte_univ}): {len(frecuentes_univ_ap)}\n")
    out.write(f"Total Reglas generadas (Lift > 1.0): {len(reglas_univ_ap)}\n\n")
    if not reglas_univ_ap.empty:
        reglas_univ_ap['antecedents_str'] = reglas_univ_ap['antecedents'].apply(lambda x: ', '.join(list(x)))
        reglas_univ_ap['consequents_str'] = reglas_univ_ap['consequents'].apply(lambda x: ', '.join(list(x)))
        top_rules = reglas_univ_ap.sort_values(by='lift', ascending=False).head(35)
        for _, row in top_rules.iterrows():
            out.write(f"{row['antecedents_str']} -> {row['consequents_str']} | Sup: {row['support']:.3f} | Conf: {row['confidence']:.2%} | Lift: {row['lift']:.2f}\n")
