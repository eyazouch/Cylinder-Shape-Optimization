# Optimisation de la forme d'un récipient cylindrique

## Description
Ce projet étudie l'optimisation de la forme d'un récipient cylindrique pour minimiser la quantité de matériaux utilisés tout en maintenant un volume constant. Il s'agit d'un problème classique d'optimisation non linéaire avec des applications pratiques dans l'industrie de l'emballage, la conception de contenants, et le stockage.

## Contenu du projet
- **`cylinder_optimization.py`** : Script Python qui résout le problème d'optimisation et génère les visualisations
- **`rapport.pdf`** : Rapport détaillé présentant la formulation mathématique, la résolution et l'analyse des résultats
- **`figures/`** : Dossier contenant les visualisations générées

## Résultats principaux
L'étude démontre mathématiquement que la forme optimale d'un cylindre est obtenue lorsque la hauteur est égale à deux fois le rayon (h = 2r). Cette configuration minimise la surface totale du cylindre pour un volume donné.

## Visualisations
Le projet inclut plusieurs visualisations qui illustrent le problème et sa solution :
- Courbe de la surface en fonction du rapport hauteur/rayon
- Comparaison 3D de différentes formes de cylindres (plat, carré, optimal, allongé)
- Représentation de la surface d'optimisation en 3D

## Technologies utilisées
- Python 3.x
- Bibliothèques : NumPy, Matplotlib, SciPy

## Installation et exécution
```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/optimisation-cylindre.git
cd optimisation-cylindre

# Installer les dépendances
pip install numpy matplotlib scipy

# Exécuter le script principal
python cylinder_optimization.py
```

## Contexte académique
Ce projet a été réalisé dans le cadre du cours d'optimisation non linéaire à la faculté des sciences de Sfax. Il illustre l'application des méthodes d'optimisation à un problème d'ingénierie concret.


---

*Note: Ce projet démontre comment une question apparemment simple peut cacher une élégance mathématique, et comment sa résolution peut générer des économies matérielles significatives à l'échelle industrielle.*
