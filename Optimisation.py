# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1S9NxXe4Kwhrq2dCvra1z5yJxP_Gfmzov
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import minimize

# Fonction pour calculer la surface d'un cylindre
def surface_cylindre(r, h):
    """Calcule la surface totale d'un cylindre de rayon r et hauteur h"""
    return 2 * np.pi * r**2 + 2 * np.pi * r * h

# Fonction pour calculer le volume d'un cylindre
def volume_cylindre(r, h):
    """Calcule le volume d'un cylindre de rayon r et hauteur h"""
    return np.pi * r**2 * h

# Fonction objective: surface à minimiser
def objective(variables):
    r, h = variables
    return surface_cylindre(r, h)

# Contrainte: volume fixe
def constraint(variables, volume_cible):
    r, h = variables
    return volume_cylindre(r, h) - volume_cible

# Optimisation directe avec scipy
def optimiser_cylindre(volume_cible=1.0):
    """Optimise la forme d'un cylindre pour minimiser la surface avec un volume fixe"""
    # Estimation initiale: cube (r=h/2)
    initial_guess = [0.5, 2.0]

    # Contrainte: volume constant
    constraint_dict = {
        'type': 'eq',
        'fun': constraint,
        'args': (volume_cible,)
    }

    # Bornes: r > 0, h > 0
    bounds = [(0.01, None), (0.01, None)]

    # Optimisation
    result = minimize(
        objective,
        initial_guess,
        bounds=bounds,
        constraints=constraint_dict,
        method='SLSQP'
    )

    return result.x, result.fun

# Fonction pour afficher le cylindre 3D
def afficher_cylindre_3d(r, h, ax=None, color='blue', alpha=0.3):
    """Affiche un cylindre 3D de rayon r et hauteur h"""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Points pour tracer le cylindre
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, h, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = r * np.cos(theta_grid)
    y_grid = r * np.sin(theta_grid)

    # Surface latérale
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha)

    # Bases circulaires
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = r * np.cos(theta)
    y_circle = r * np.sin(theta)

    ax.plot(x_circle, y_circle, np.zeros_like(theta), color=color)
    ax.plot(x_circle, y_circle, np.ones_like(theta) * h, color=color)

    # Configurer les axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax

# Visualiser la relation surface-volume pour différentes formes
def visualiser_relation_forme_surface():
    """
    Visualise comment la surface varie en fonction du rapport hauteur/rayon
    pour un volume constant
    """
    volume_cible = 1.0

    # Différents rapports hauteur/rayon
    ratios = np.logspace(-1, 1, 100)  # de 0.1 à 10
    surfaces = []

    for ratio in ratios:
        # h/r = ratio => h = ratio*r
        # Volume = π*r²*h = π*r²*ratio*r = π*ratio*r³
        # => r = (Volume/(π*ratio))^(1/3)
        r = (volume_cible / (np.pi * ratio))**(1/3)
        h = ratio * r
        surfaces.append(surface_cylindre(r, h))

    # Point optimal
    r_opt, h_opt = optimiser_cylindre(volume_cible)[0]
    ratio_opt = h_opt / r_opt
    surface_opt = surface_cylindre(r_opt, h_opt)

    plt.figure(figsize=(10, 6))
    plt.plot(ratios, surfaces, 'b-', linewidth=2)
    plt.scatter([ratio_opt], [surface_opt], color='red', s=100, zorder=5)

    plt.axvline(x=2, color='green', linestyle='--', label='h/r = 2 (optimal)')
    plt.text(2.1, surface_opt + 0.5, 'h/r = 2\nDimensionnement optimal',
             fontsize=12, color='green')

    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Rapport hauteur/rayon (h/r)', fontsize=12)
    plt.ylabel('Surface totale', fontsize=12)
    plt.title('Surface en fonction du rapport hauteur/rayon (volume = 1)', fontsize=14)

    # Annoter quelques points spécifiques
    annotations = [
        (0.2, 'Disque plat\n(h << r)'),
        (10, 'Tube fin\n(h >> r)')
    ]

    for ratio, text in annotations:
        idx = np.abs(ratios - ratio).argmin()
        plt.annotate(text,
                     xy=(ratios[idx], surfaces[idx]),
                     xytext=(ratios[idx]*1.2, surfaces[idx]*1.1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=12)

    plt.tight_layout()
    plt.show()

# Comparer différentes formes de cylindres
def comparer_cylindres():
    """Compare visuellement différentes formes de cylindres avec le même volume"""
    volume_cible = 1.0

    # Obtenir la forme optimale
    (r_opt, h_opt), surface_opt = optimiser_cylindre(volume_cible)

    # Créer d'autres formes avec le même volume
    ratios = [0.2, 1.0, 2.0, 5.0]  # h/r
    cylindres = []

    for ratio in ratios:
        r = (volume_cible / (np.pi * ratio))**(1/3)
        h = ratio * r
        surface = surface_cylindre(r, h)
        cylindres.append((r, h, surface, ratio))

    # Configuration de la figure
    fig = plt.figure(figsize=(15, 10))

    # Afficher les cylindres côte à côte
    labels = ["Très plat\n(h/r = 0.2)", "Carré\n(h/r = 1)",
              "Optimal\n(h/r = 2)", "Allongé\n(h/r = 5)"]
    colors = ['#FFA07A', '#98FB98', '#87CEFA', '#FFFF99']

    for i, (r, h, surface, ratio) in enumerate(cylindres):
        # Sous-graphique pour chaque cylindre
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        afficher_cylindre_3d(r, h, ax, color=colors[i])

        # Titre et annotations
        ax.set_title(f"{labels[i]}\nSurface: {surface:.2f}", fontsize=12)

        # Égaliser les échelles
        max_dim = max(r*2, h)
        ax.set_xlim([-max_dim/2, max_dim/2])
        ax.set_ylim([-max_dim/2, max_dim/2])
        ax.set_zlim([0, max_dim])

        # Mettre en évidence le cylindre optimal
        if abs(ratio - 2.0) < 0.1:
            ax.set_title(f"{labels[i]}\nSurface: {surface:.2f} (MINIMUM)",
                        fontsize=12, color='green', fontweight='bold')

    plt.suptitle('Comparaison de cylindres de même volume (V = 1) avec différents rapports h/r',
                fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Visualiser la surface en 3D
def visualiser_surface_3d():
    """Visualise la surface comme une fonction de r et h, avec la contrainte de volume"""
    volume_cible = 1.0

    # Grille de rayons
    r_range = np.linspace(0.2, 2.0, 50)

    # Grille de surfaces
    surf_values = np.zeros((len(r_range), len(r_range)))
    h_values = np.zeros((len(r_range), len(r_range)))

    # Pour chaque rayon, calculer la hauteur correspondante pour maintenir le volume
    # et calculer la surface résultante
    for i, r in enumerate(r_range):
        for j, r2 in enumerate(r_range):
            h = volume_cible / (np.pi * r2**2)
            h_values[i, j] = h
            surf_values[i, j] = surface_cylindre(r2, h)

    # Point optimal
    r_opt, h_opt = optimiser_cylindre(volume_cible)[0]
    surface_opt = surface_cylindre(r_opt, h_opt)

    # Créer la figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Créer une grille 2D pour r et h
    r_grid, h_grid = np.meshgrid(r_range, r_range)

    # Tracer la surface 3D
    surf = ax.plot_surface(r_grid, h_values, surf_values, cmap=cm.viridis,
                          linewidth=0, antialiased=True, alpha=0.8)

    # Marquer le point optimal
    ax.scatter([r_opt], [h_opt], [surface_opt], color='red', s=100, label='Point optimal')

    # Configurer les axes
    ax.set_xlabel('Rayon (r)')
    ax.set_ylabel('Hauteur (h)')
    ax.set_zlabel('Surface')
    ax.set_title('Surface du cylindre en fonction de r et h avec volume constant (V = 1)')

    # Projeter la trajectoire du point optimal
    h_constrainte = volume_cible / (np.pi * r_range**2)
    surface_constrainte = surface_cylindre(r_range, h_constrainte)
    ax.plot(r_range, h_constrainte, surface_constrainte, 'r-', linewidth=3,
            label='Contrainte: V = 1')

    # Ajouter la position optimale (h = 2r)
    r_curve = np.linspace(0.3, 1.5, 100)
    h_curve = 2 * r_curve
    volume_curve = volume_cylindre(r_curve, h_curve)
    surface_curve = surface_cylindre(r_curve, h_curve)

    # Trouver l'indice le plus proche du volume cible
    idx_proche = np.abs(volume_curve - volume_cible).argmin()
    ax.plot(r_curve, h_curve, surface_curve, 'g-', linewidth=3,
            label='h = 2r (relation optimale)')

    # Ajouter une légende
    ax.legend()

    # Ajouter une barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Surface')

    plt.tight_layout()
    plt.show()

def visualiser_surface_rh_3d(r_min=0.1, r_max=2.0, h_min=0.1, h_max=5.0, n=50):
    """
    Trace la surface totale S(r,h) = 2πr² + 2πr h
    sur [r_min, r_max]×[h_min, h_max],
    et y superpose la courbe V=1 (h = 1/(π r²)).
    """
    # Grille de r et h pour la surface
    r = np.linspace(r_min, r_max, n)
    h = np.linspace(h_min, h_max, n)
    R, H = np.meshgrid(r, h)

    # Calcul de la surface
    S = surface_cylindre(R, H)

    # Création du plot 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(R, H, S,
                           cmap=cm.viridis,
                           edgecolor='none',
                           alpha=0.8)

    # ------------------------------------------------
    # Courbe V = 1 : h = 1/(π r²)
    # ------------------------------------------------
    r_line = np.linspace(r_min, r_max, 200)
    h_line = 1.0 / (np.pi * r_line**2)
    # filtrer pour rester dans [h_min, h_max]
    mask = (h_line >= h_min) & (h_line <= h_max)
    r_line = r_line[mask]
    h_line = h_line[mask]
    S_line = surface_cylindre(r_line, h_line)

    ax.plot(r_line, h_line, S_line,
            color='red',
            linewidth=3,
            label='V = 1 (h = 1/(π r²))')
    ax.legend()
    # ------------------------------------------------

    # Étiquettes et titre
    ax.set_xlabel('Rayon r')
    ax.set_ylabel('Hauteur h')
    ax.set_zlabel('Surface S(r,h)')
    ax.set_title('Surface totale du cylindre et coupe V=1')

    # Barre de couleur
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                 label='Surface')

    plt.tight_layout()
    plt.show()

def visualiser_volume_3d(r_min=0.1, r_max=2.0, h_min=0.1, h_max=5.0, n=50, volume_cible=1.0):
    """
    Trace la surface S(r,h)=2πr² + 2πr h
    et y superpose la feuille πr²h=1 (V=1) extrudée verticalement,
    avec leur intersection en jaune.
    """
    # Grille pour S(r,h)
    r = np.linspace(r_min, r_max, n)
    h = np.linspace(h_min, h_max, n)
    R, H = np.meshgrid(r, h)
    S = surface_cylindre(R, H)

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # 1) Surface S(r,h)
    surf = ax.plot_surface(R, H, S,
                           cmap=cm.viridis,
                           edgecolor='none',
                           alpha=0.8)

    # 2) Feuille V=1 (extrudée verticalement)
    # Ligne V=1 dans le plan r–h : h = 1/(π r²)
    r_line = np.linspace(r_min, r_max, 200)
    h_line = 1.0 / (np.pi * r_line**2)
    mask   = (h_line >= h_min) & (h_line <= h_max)
    r_line = r_line[mask]
    h_line = h_line[mask]

    # Extrusion de la ligne le long de S
    S_min, S_max = S.min(), S.max()
    s_vals = np.linspace(S_min, S_max, 20)
    R_plane, S_plane = np.meshgrid(r_line, s_vals)
    H_plane = np.tile(h_line, (len(s_vals), 1))

    ax.plot_surface(R_plane,
                    H_plane,
                    S_plane,
                    color='red',
                    alpha=0.4,
                    edgecolor='none')

    # 3) Intersection curve en noir
    S_line = surface_cylindre(r_line, h_line)
    ax.plot(r_line, h_line, S_line,
            color='black',
            linewidth=4,
            label='Intersection V=1 ⇆ S(r,h)')

    # Légendes, axes, titre
    ax.set_xlabel('Rayon r')
    ax.set_ylabel('Hauteur h')
    ax.set_zlabel('Surface S(r,h)')
    ax.set_title('Surface S(r,h) et feuille V=1 avec intersection en jaune')

    # Couleur de la surface + légende de la courbe
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Surface S')
    ax.legend()

    plt.tight_layout()
    plt.show()

    """
    Trace la surface S(r,h)=2πr² + 2πr h
    et y superpose la feuille πr²h=1 (V=1) extrudée verticalement,
    avec leur intersection en jaune.
    """
    # Grille pour S(r,h)
    r = np.linspace(r_min, r_max, n)
    h = np.linspace(h_min, h_max, n)
    R, H = np.meshgrid(r, h)
    S = surface_cylindre(R, H)

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # 1) Surface S(r,h)
    surf = ax.plot_surface(R, H, S,
                           cmap=cm.viridis,
                           edgecolor='none',
                           alpha=0.8)

    # 2) Feuille V=1 (extrudée verticalement)
    # Ligne V=1 dans le plan r–h : h = 1/(π r²)
    r_line = np.linspace(r_min, r_max, 200)
    h_line = 1.0 / (np.pi * r_line**2)
    mask   = (h_line >= h_min) & (h_line <= h_max)
    r_line = r_line[mask]
    h_line = h_line[mask]

    # Extrusion de la ligne le long de S
    S_min, S_max = S.min(), S.max()
    s_vals = np.linspace(S_min, S_max, 20)
    R_plane, S_plane = np.meshgrid(r_line, s_vals)
    H_plane = np.tile(h_line, (len(s_vals), 1))

    ax.plot_surface(R_plane,
                    H_plane,
                    S_plane,
                    color='red',
                    alpha=0.4,
                    edgecolor='none')

    # 3) Intersection curve en jaune
    S_line = surface_cylindre(r_line, h_line)
    ax.plot(r_line, h_line, S_line,
            color='yellow',
            linewidth=4,
            label='Intersection V=1 ⇆ S(r,h)')

    # Légendes, axes, titre
    ax.set_xlabel('Rayon r')
    ax.set_ylabel('Hauteur h')
    ax.set_zlabel('Surface S(r,h)')
    ax.set_title('Surface S(r,h) et feuille V=1 avec intersection en jaune')

    # Couleur de la surface + légende de la courbe
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Surface S')
    ax.legend()

    plt.tight_layout()
    plt.show()



# Fonction principale
def main():
    # Volume cible
    volume_cible = 5.0

    # Optimiser la forme
    (r_opt, h_opt), surface_opt = optimiser_cylindre(volume_cible)

    print(f"Résultats de l'optimisation pour un volume de {volume_cible}:")
    print(f"Rayon optimal: {r_opt:.4f}")
    print(f"Hauteur optimale: {h_opt:.4f}")
    print(f"Rapport h/r: {h_opt/r_opt:.4f}")
    print(f"Surface minimale: {surface_opt:.4f}")

    # Comparaison avec d'autres formes
    r_carre = (volume_cible / np.pi)**(1/3)
    h_carre = r_carre
    surface_carre = surface_cylindre(r_carre, h_carre)

    r_plat = (volume_cible / (0.2 * np.pi))**(1/3)
    h_plat = 0.2 * r_plat
    surface_plat = surface_cylindre(r_plat, h_plat)

    r_allonge = (volume_cible / (5 * np.pi))**(1/3)
    h_allonge = 5 * r_allonge
    surface_allonge = surface_cylindre(r_allonge, h_allonge)

    print("\nComparaison des surfaces pour différentes formes (même volume):")
    print(f"Cylindre optimal (h/r = 2):     Surface = {surface_opt:.4f}")
    print(f"Cylindre 'carré' (h/r = 1):     Surface = {surface_carre:.4f} (+{(surface_carre/surface_opt-1)*100:.2f}%)")
    print(f"Cylindre plat (h/r = 0.2):      Surface = {surface_plat:.4f} (+{(surface_plat/surface_opt-1)*100:.2f}%)")
    print(f"Cylindre allongé (h/r = 5):     Surface = {surface_allonge:.4f} (+{(surface_allonge/surface_opt-1)*100:.2f}%)")

    # Visualiser les résultats
    print("\nCréation des visualisations...")

    print("1. Relation entre le rapport hauteur/rayon et la surface...")
    visualiser_relation_forme_surface()

    print("2. Comparaison des différentes formes de cylindres...")
    comparer_cylindres()

    print("3. Visualisation de la surface en 3D...")
    visualiser_surface_3d()

    print("4. Visualisation de S(r,h) sans contrainte de volume...")
    visualiser_surface_rh_3d()

    print("5. Surface V(r,h) et plan V=1 en rouge...")
    visualiser_volume_3d()


if __name__ == "__main__":
    main()
