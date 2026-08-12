# --- sequence_logo.py ---
"""
This script contains tools to plot a sequence logo
from an evomd csv report.
"""

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from sequence import Sequence


def normalizar_array(arr, min_target=0.05, max_target=1.0):
    if not arr:
        return []
    
    # Se obtienen el máximo y mínimo sin importar el orden del array
    max_val = max(arr)
    min_val = min(arr)
    rango_origen = max_val - min_val

    # Si todos los elementos son iguales, se devuelve el valor máximo
    if rango_origen == 0:
        return [max_target] * len(arr)

    rango_destino = max_target - min_target

    return [min_target + ((x - min_val) * rango_destino / rango_origen) for x in arr]


def plot_sequence_logo(sequences, color='cornflowerblue', outline=True, save=False, out='sequencelogo.png', exclude: (str|list|None) = None):
    """
    Genera un sequence logo de frecuencias relativas (0 a 1) usando solo matplotlib.
    
    Parámetros:
    - sequences: Lista de strings con las secuencias.
    - color: Color del relleno de las letras.
    - outline: Booleano (True/False). Activa o desactiva la línea negra delgada de contorno.
    - save: Booleano (True/False). Guarda la figura en un archivo.
    - out: Nombre del archivo de salida.
    - exclude: letters to be excluded
    """
    if not sequences:
        print("La lista de secuencias está vacía.")
        return

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
    else:
        exclude = []
        
    num_seqs = len(sequences)
    seq_len = len(sequences[0])
    
    # 1. Calcular frecuencias relativas por posición
    freqs_per_pos = []
    for i in range(seq_len):
        columna = [seq[i] for seq in sequences if seq[i] not in exclude]
        conteos = Counter(columna)
        # frecuencias = {aa: count / num_seqs for aa, count in conteos.items()}
        frecuencias = {aa: count / len(columna) for aa, count in conteos.items()}
        
        # MODIFICACIÓN: reverse=True para que las frecuencias altas queden en la base
        frecuencias_ordenadas = sorted(frecuencias.items(), key=lambda x: x[1], reverse=True)
        freqs_per_pos.append(frecuencias_ordenadas)
        
    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.5), 4))
    
    # 2. Configuración de fuente
    funente_path = "fonts/SilkRemington-SBold.ttf"
    propiedades_fuente = FontProperties(fname=funente_path)
    
    # 3. Solución "bolígrafo": Calcular escala X global basada en la letra más ancha
    letras_posibles = "ACDEFGHIKLMNPQRSTVWY"
    anchos = []
    for letra in letras_posibles:
        tp = TextPath((0, 0), letra, size=1, prop=propiedades_fuente)
        anchos.append(tp.get_extents().width)
    max_width = max(anchos) if anchos else 1.0
    
    # Fijamos el factor de escala X para todas las letras (0.8 es el ancho máximo por columna)
    scale_x_global = 0.8 / max_width
    
    # Configuración del contorno removible
    edge_color = 'black' if outline else 'none'
    line_width = 0.5 if outline else 0

    # 4. Dibujar las letras
    for i, pos_freqs in enumerate(freqs_per_pos):
        x_pos = i + 1
        y_offset = 0.0
        alphas = normalizar_array([k[1] for k in pos_freqs])

        a = 0
        for aa, freq in pos_freqs:
            if freq == 0: 
                continue
            
            tp = TextPath((0, 0), aa, size=1, prop=propiedades_fuente)
            bbox = tp.get_extents()
            
            if bbox.width == 0 or bbox.height == 0:
                continue
            
            # La escala Y sigue dependiendo de la frecuencia relativa
            scale_y = freq / bbox.height
            
            # Centrar la letra horizontalmente en su columna
            width_escalado = bbox.width * scale_x_global
            posicion_x_centro = x_pos - (width_escalado / 2)
            
            transformacion = Affine2D() \
                .scale(scale_x_global, scale_y) \
                .translate(posicion_x_centro - (bbox.x0 * scale_x_global), y_offset - (bbox.y0 * scale_y))
            
            tp_transformado = tp.transformed(transformacion)
            
            # Añadimos el parche aplicando los contornos requeridos
            patch = PathPatch(tp_transformado, facecolor=color, edgecolor=edge_color, linewidth=line_width, alpha=alphas[a])
            ax.add_patch(patch)
            
            y_offset += freq
            a += 1

    # 5. Formato de los ejes
    ax.set_xlim(0.5, seq_len + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, seq_len + 1))
    
    ax.set_ylabel('Relative frequency')
    ax.set_xlabel('Section')
    
    # Limpiar bordes superior y derecho
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save:
        plt.savefig(out, dpi=300)
    plt.show()


def main():
    pass

if __name__ == '__main__':
    main()
