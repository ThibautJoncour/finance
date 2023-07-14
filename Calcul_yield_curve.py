import numpy as np
import numpy as np
import matplotlib.pyplot as plt

class zero_coupons:
    
      # Faire le calcul matriciel à partir des obligations zéros coupons
    
    def __init__(self,matrice_flux,vecteur_prix):
        self.matrice_flux = matrice_flux
        self.vecteur_prix = vecteur_prix
        
    def calcul_vecteur_facteur_actualisation(self):
        matrice_inverse = np.linalg.inv(self.matrice_flux)

        vecteur = self.vecteur_prix
        matrice_inverse = np.linalg.inv(self.matrice_flux)
        return np.dot(matrice_inverse, vecteur)


class calcul_yield_curve:
    
    # Faire le calcul du vecteur d'actualisation
  
    def __init__(self, vecteur_actualisation):
        self.vecteur_actualisation = vecteur_actualisation
        
    
    def __repr__(self):
        return f"{self.vecteur_actualisation}"

class graphique:

    # Faire le graphique de la courbe des taux
    
    def graph(self, taux):
        self.taux = taux
        periodes = np.arange(1, len(self.taux) + 1)
        plt.plot(periodes, taux)

        plt.xlabel('Maturité')
        plt.ylabel('Taux %')
        plt.title('Courbe des taux')
        plt.show()
        
    
    
    
# Considérons 4 obligations
# Obligation 1 : offrant des coupons de 5% remboursement de 105 à écheance de 2 ans ayant un prix de 91.16
# Obligation 2 : offrant des coupons de 6% remboursement de 106 à écheance de 2 ans ayant un prix de 96.59
# Obligation 3 : offrant des coupons de 5% remboursement de 105 à écheance de 3 ans ayant un prix de 88
# Obligation 4 : Zéro coupon, offrant un remboursement de 105 à écheance de 4 ans ayant un prix de 70.5

# O
matrice_flux = np.array([[3,103,0,0],[6,106,0,0],[5,5,105,0],[0,0,0,105]])
vecteur_prix = np.array([91.16, 96.59, 88, 70.5])





vecteur_actualisation = zero_coupons(matrice_flux, vecteur_prix)
vect  = (vecteur_actualisation.calcul_vecteur_facteur_actualisation())

yield_curve = calcul_yield_curve(vect)

# Extraire les taux à partir du vecteur d'actualisation


taux = []
for i in range(1, len(vect)+1):
    taux.append((1/vect[i-1])**(1/i) - 1)


taux = (np.array(taux)*100)
# Graphique de la courbe des taux


graph = graphique()
graph.graph(taux)

