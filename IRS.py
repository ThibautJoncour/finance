#!/usr/bin/env python
# coding: utf-8

# In[219]:


from datetime import datetime
from dateutil import relativedelta
import numpy as np
import pandas as pd

# get two dates
d1 = '30/9/2011'
d2 = '31/12/2016'
d3 = '30/9/2017'


# convert string to date object
start_date = datetime.strptime(d1, "%d/%m/%Y")
date_valo = datetime.strptime(d2, "%d/%m/%Y")
end_date = datetime.strptime(d3, "%d/%m/%Y")

# Get the relativedelta between two dates

delta2 = relativedelta.relativedelta(end_date, start_date)
print('Durée totale du swap')
print(delta2.years, 'Years,', delta2.months, 'months,', delta2.days, 'days')

delta = relativedelta.relativedelta(end_date, date_valo)
print('Durée residuelle du swap')
print(delta.years, 'Years,', delta.months, 'months,', delta.days, 'days')


# In[222]:



date_demission = '2011-09'

coupons_date = []
durée_swap = 6
for i in range(6,durée_swap*(12)+6,6):
    print('la date du prochain coupon semestriel est le',np.datetime64(date_demission) + np.timedelta64(i, 'M'))
    coupons_date.append(np.datetime64(date_demission) + np.timedelta64(i, 'M'))
coupons_date = pd.DataFrame(coupons_date)
# Coupons restants à obtenir à la date de valorisation
coupons_restant = coupons_date[coupons_date[0]>date_valo]
print('il reste',len(coupons_restant),'coupons à obtenir au',date_valo)


# In[150]:


nbre_coupons_restant = len(coupons_restant)
taux = [0.05,0.06,0.07,0.08]


# In[223]:


class pricing_jf:

	def __init__(self,notionnel):
		self.trois_mois = taux[0]
		self.six_mois = taux[1]
		self.neuf_mois = taux[2]
		self.douze_mois = taux[3]
		self.taux_fixe = 5/100
		self.coupons = notionnel* self.taux_fixe
		self.jambe_fixe = (self.coupons/((1+self.trois_mois*(3/12))))+(self.coupons/((1+self.neuf_mois*(9/12))))

		return

class pricing_jv:

	def __init__(self,notionnel,taux_forward):
		self.trois_mois = taux[0]
		self.six_mois = taux[1]
		self.neuf_mois = taux[2]
		self.douze_mois = taux[3]
		self.taux_libor = 6/100
		self.coupons_variable_t = (notionnel* self.taux_libor)/2
		self.coupons_variable_fwrd = (notionnel* (taux_forward/100))/2
		self.jambe_variable = (self.coupons_variable_t/((1+self.trois_mois*(3/12))))+(self.coupons_variable_fwrd/((1+self.neuf_mois*(9/12))))
		return

taux_fwrd = []
def calcul_taux_forward(t1,t2,d1,d2):
    taux_fwrd.append((((((1+t2)**d2)/(1+t1)**d1)**(1/(d2-d1))-1))*100)
    print('Le taux forward à', d2,'mois est de:',str(taux_fwrd)[1:5],'%')
    return taux_fwrd

swap = pricing_jf(1000000)
taux_forward = calcul_taux_forward(swap.trois_mois,swap.neuf_mois,3,9)


# In[224]:


jv = pricing_jv(1000000,np.array(taux_forward))
print('La valeur de la jambe variable au',date_valo,' est de:',str(jv.jambe_variable)[1:-1],'€')
jf = pricing_jf(1000000)
print('La valeur de la jambe fixe au',date_valo,' est de:',jf.jambe_fixe,'€')


# In[225]:


IRS_receveur = jf.jambe_fixe - jv.jambe_variable
IRS_payeur = -IRS_receveur

print('La valeur du SWAP receveur au',date_valo,'est de:',IRS_receveur,'€','et la valeur du SWAP payeur est de',IRS_payeur)


# In[ ]:




