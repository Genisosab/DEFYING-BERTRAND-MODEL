from sympy import symbols, Eq, solve
import sympy as sp
from scipy import optimize, arange
from numpy import array
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import pylab as plt
from operator import itemgetter
from time import time
from scipy.interpolate import make_interp_spline

start2 = time()

q = sp.Symbol('q')
costes_totales = 100 + q**2 -10*q
costes_marginales = sp.diff(costes_totales, q)
costes_totales_medios = (costes_totales)/q

ecuacion_equilibrio = costes_marginales - costes_totales_medios

solucion_cantidad_equilibrio = solve(ecuacion_equilibrio)
cantidad_minima = solucion_cantidad_equilibrio[1]

q = cantidad_minima
costes_totales = 100 + q**2 -10*q

p = sp.Symbol('p')
ecuacion_precio = p*cantidad_minima - costes_totales
precio_minimo = solve(ecuacion_precio)[0]

costes_totales_medios_por_cantidad =[]
costes_marginales_por_cantidad = []
cantidades = []

q = sp.Symbol('q')
costes_totales = 100 + q**2 -10*q
coste_marginal = sp.diff(costes_totales, q)

i = 100

coste_marginal = coste_marginal.subs(q, i)

for i in range(100):

    q = sp.Symbol('q')
    costes_totales = 100 + q**2 -10*q
    coste_marginal = sp.diff(costes_totales, q)  

    q = i
    costes_totales = 100 + q**2 -10*q
    
    if i == 0:
        pass
    else: 
        costes_totales_medios = (costes_totales)/q
        q = sp.Symbol('q')
        coste_marginal = coste_marginal.subs(q, i)         
        cantidades.append(i)
        costes_marginales_por_cantidad.append(coste_marginal)
        costes_totales_medios_por_cantidad.append(costes_totales_medios)

plt.plot(cantidades, costes_marginales_por_cantidad)
plt.plot(cantidades, costes_totales_medios_por_cantidad)
plt.plot(cantidad_minima, precio_minimo, "ro")
plt.show()

class consumidor:

    def __init__(self, numero, renta):
        self.numero = numero
        self.renta = renta

    def __str__(self):
        return '\nConsumidor: ' + str(self.numero) + '\nRenta: ' + str(self.renta)

numero_consumidores = 200

lista_consumidor = []

for i in range(numero_consumidores):
    rentax = random.randint(500,2000)
    consumidorx = consumidor(i+1, rentax)
    lista_consumidor.append(rentax)

lista_consumidor_ordenada_por_renta = sorted(lista_consumidor)

beneficios_obtenidos_por_precio = []
precio_maximo = max(lista_consumidor_ordenada_por_renta)

for j in range(precio_minimo, precio_maximo):
    cantidad_total_vendida = 0
    for i in range(numero_consumidores):
        if lista_consumidor_ordenada_por_renta[i] < j:
            pass
        else:
            cantidad_vendida = math.floor(lista_consumidor_ordenada_por_renta[i]/j)

            cantidad_total_vendida = cantidad_total_vendida + cantidad_vendida

            if i == len(lista_consumidor_ordenada_por_renta)-1:
                beneficios_obtenidos_por_precio.append([j, cantidad_total_vendida])

beneficio_nash = beneficios_obtenidos_por_precio[0][1]

costes_totales = 100 + q**2 -10*q
costes_totales_medios = costes_totales/q

lista_beneficios_posibles = []
lista_beneficios_posibles_segun_precio_cantidad = []
lista_beneficios_cantidad = []
lista_beneficios_precio = []

for i in range(len(beneficios_obtenidos_por_precio)):
    p = beneficios_obtenidos_por_precio[i][0]
    q = beneficios_obtenidos_por_precio[i][1]
    ingresos = p*q
    costes = costes_totales = 100 + q**2 -10*q
    beneficios = ingresos - costes
    if beneficios >= 0:
        lista_beneficios_posibles.append(beneficios)
        lista_beneficios_cantidad.append(q)
        lista_beneficios_precio.append(p)
        lista_beneficios_posibles_segun_precio_cantidad.append([p, q, beneficios])

beneficio_maximo = max(lista_beneficios_posibles)

for i in range(len(lista_beneficios_posibles_segun_precio_cantidad)):
    if lista_beneficios_posibles_segun_precio_cantidad[i][2] == beneficio_maximo:
        cantidad_beneficio_maximo = lista_beneficios_posibles_segun_precio_cantidad[i][1]
        precio_beneficio_maximo = lista_beneficios_posibles_segun_precio_cantidad[i][0]
    else:
        pass

X_Y_Spline = make_interp_spline(lista_beneficios_precio, lista_beneficios_posibles)
X_ = np.linspace(min(lista_consumidor_ordenada_por_renta), max(lista_consumidor_ordenada_por_renta), 30)
Y_ = X_Y_Spline(X_)

#plt.plot(lista_beneficios_cantidad, lista_beneficios_posibles)
#plt.plot(lista_beneficios_precio, lista_beneficios_posibles)
#plt.plot(lista_beneficios_cantidad, lista_beneficios_precio)

plt.plot(X_, Y_)

plt.plot(precio_beneficio_maximo, beneficio_maximo, "ro")
plt.axvline(x=precio_beneficio_maximo, color='red', linestyle='--')

plt.xlabel('Precio')
plt.ylabel('Beneficio')

plt.show()

lista_estados = []

for i in range(len(lista_beneficios_posibles_segun_precio_cantidad)):
    if lista_beneficios_posibles_segun_precio_cantidad[i][0] <= precio_beneficio_maximo:
        lista_estados.append(lista_beneficios_posibles_segun_precio_cantidad[i][0])

episodios = 300000
empresas = 2

lista_consumidor_episodio = []

q_tables = []
for i in range(empresas):
    matriz = np.zeros((len(lista_estados), 6))
    q_tables.append([matriz])

lista_beneficios=[]

lista_acciones_numericas = []

lista_numero_episodios = []
lista_evolucion_precios_episodio = []

lista_cuotas = [1, (1/empresas)]
lista_episodios_convergentes = []
accion = 0

for i in range(episodios):
    tasa_aprendizaje = 0.99 - 0.000005*i
    if tasa_aprendizaje <= 0.01:
        tasa_aprendizaje = 0.01
    factor_descuento = 0.95
    lista_numero_episodios.append(i)
    lista_consumidor_episodio = lista_consumidor_ordenada_por_renta.copy()  
    lista_acciones = []
    lista_estados_empresa = []
    lista_precios_por_empresa = []
    
    for j in range(empresas):

        if i == 0:
            estado_anterior = random.choice(lista_estados)
            estado_actual = random.choice(lista_estados)
            cuota = random.choice(lista_cuotas)
            if estado_anterior < estado_actual and cuota== 1: #aumentar con cantidad 100%
                accion = 0
            if estado_anterior > estado_actual and cuota== 1: #disminuir con cantidad 100%
                accion = 1
            if estado_anterior == estado_actual and cuota== 1: #mantener con cantidad 100%
                accion = 2
            if estado_anterior < estado_actual and cuota== (1/empresas): #aumentar con cantidad (1/empresas)
                accion = 3
            if estado_anterior > estado_actual and cuota== (1/empresas): #disminuir con cantidad (1/empresas)
                accion = 4
            if estado_anterior == estado_actual and cuota== (1/empresas): #mantener con cantidad (1/empresas)
                accion = 5
        else:
            max_q_greedy = np.max(q_tables[j][0])

            if random.random() <= tasa_aprendizaje:

                estado_anterior = estado_actual
                estado_actual = random.choice(lista_estados)

                cuota = random.choice(lista_cuotas)
                if estado_anterior < estado_actual and cuota== 1: #aumentar con cantidad 100%
                    accion = 0
                if estado_anterior > estado_actual and cuota== 1: #disminuir con cantidad 100%
                    accion = 1
                if estado_anterior == estado_actual and cuota== 1: #mantener con cantidad 100%
                    accion = 2
                if estado_anterior < estado_actual and cuota== (1/empresas): #aumentar con cantidad (1/empresas)
                    accion = 3
                if estado_anterior > estado_actual and cuota== (1/empresas): #disminuir con cantidad (1/empresas)
                    accion = 4
                if estado_anterior == estado_actual and cuota== (1/empresas): #mantener con cantidad (1/empresas)
                    accion = 5
            else:

                for x in range(len(lista_estados)):
                    for h in range(6):
                        if q_tables[j][0][x, h] == max_q_greedy:
                            estado_anterior = estado_actual
                            estado_actual = x
                            if estado_anterior < estado_actual: #aumentar precio con cantidad 100%
                                accion = 0
                            if estado_anterior > estado_actual: #disminuir precio con cantidad 100%
                                accion = 1
                            if estado_anterior == estado_actual: #mantener precio con cantidad 100%
                                accion = 2
                            if estado_anterior < estado_actual: #aumentar precio con cantidad (1/empresas)
                                accion = 3
                            if estado_anterior > estado_actual: #disminuir precio con cantidad (1/empresas)
                                accion = 4
                            if estado_anterior == estado_actual: #mantener precio con cantidad (1/empresas)
                                accion = 5

        lista_acciones.append(accion)
        lista_estados_empresa.append([estado_anterior, estado_actual])
        lista_precios_por_empresa.append([j, estado_actual])
        lista_evolucion_precios_episodio.append([j, estado_actual])

        lista_precios_ordenada_por_precio = sorted(lista_precios_por_empresa, key=lambda x: x[1])

    if accion == 0 or accion == 1 or accion == 2:
        for j in range(empresas):
            for c in range(len(lista_beneficios_posibles_segun_precio_cantidad)):
                if lista_precios_ordenada_por_precio[j][1] == lista_beneficios_posibles_segun_precio_cantidad[c][0]:
                    cantidad = math.floor(lista_beneficios_posibles_segun_precio_cantidad[c][1])
                else:
                    pass
            lista_precios_ordenada_por_precio[j].append(cantidad)
    else:
        for j in range(empresas):
            for c in range(len(lista_beneficios_posibles_segun_precio_cantidad)):
                if lista_precios_ordenada_por_precio[j][1] == lista_beneficios_posibles_segun_precio_cantidad[c][0]:
                    cantidad = math.floor(lista_beneficios_posibles_segun_precio_cantidad[c][1]/(empresas))
                else:
                    pass

            lista_precios_ordenada_por_precio[j].append(cantidad)

    for j in range(empresas):
        lista_beneficios.append([])
        
        cantidad_vendida_total = 0
        cantidad_ofertada = lista_precios_ordenada_por_precio[j][2]
        
        for a in range(len(lista_consumidor_episodio)):

            if cantidad_vendida_total >= lista_precios_ordenada_por_precio[j][2]:
                pass
            else:                
                
                precio_ofertado = lista_precios_ordenada_por_precio[j][1]
                
                if precio_ofertado == 0:
                    pass
                else:
                
                    cantidad_consumidor_dispuesto_compra = math.floor(lista_consumidor_episodio[a]/precio_ofertado)
                    
                    if cantidad_ofertada >= cantidad_consumidor_dispuesto_compra:
                        
                        cantidad_comprada = cantidad_consumidor_dispuesto_compra
                        lista_consumidor_episodio[a] = lista_consumidor_episodio[a] - (cantidad_comprada*precio_ofertado)
                        cantidad_ofertada = cantidad_ofertada - cantidad_comprada
                        cantidad_vendida_total = cantidad_vendida_total + cantidad_comprada

                    else:
                        cantidad_comprada = cantidad_ofertada
                        lista_consumidor_episodio[a] = lista_consumidor_episodio[a] - (cantidad_comprada*precio_ofertado)
                        cantidad_ofertada = cantidad_ofertada - cantidad_comprada
                        cantidad_vendida_total = cantidad_vendida_total + cantidad_comprada
        
        ingresos_empresa = lista_precios_ordenada_por_precio[j][1] * cantidad_vendida_total
        costes_empresa = 100 + lista_precios_ordenada_por_precio[j][2]**2 -10*lista_precios_ordenada_por_precio[j][2]
        beneficios_empresa =  ingresos_empresa - costes_empresa
        lista_precios_ordenada_por_precio[j].append(beneficios_empresa)
        lista_beneficios[j].append(beneficios_empresa)
            
    for j in range(empresas):
        lista_precios_ordenada_por_precio = sorted(lista_precios_ordenada_por_precio)

        for g in range(len(lista_estados)):
        
            fila_proximo_q = 0
            fila_actual_q = 0

            if lista_estados[g] == lista_precios_ordenada_por_precio[j][1]:
                fila_proximo_q = g
                max_q = np.max(q_tables[j][0][fila_proximo_q])

            if lista_estados[g] == lista_estados_empresa[0][1]:
                fila_actual_q = g
                valor_q_actual = q_tables[j][0][fila_actual_q][lista_acciones[j]]

                valor_q_actualizado = valor_q_actual + tasa_aprendizaje * (lista_precios_ordenada_por_precio[j][3] + factor_descuento - valor_q_actual)

                q_tables[j][0][fila_actual_q][lista_acciones[j]] = valor_q_actualizado
                convergencia = abs(valor_q_actual - valor_q_actualizado)
    if convergencia <= 50:

        lista_episodios_convergentes.append(1)
    else:
        lista_episodios_convergentes.append(0)

lista_precios_evolucion_empresa = []
lista_precios = []

for j in range(empresas):
    lista_precios.append([])

for i in range(len(lista_evolucion_precios_episodio)):
    for j in range(empresas):
        if lista_evolucion_precios_episodio[i][0] == j:
            lista_precios[j].append(lista_evolucion_precios_episodio[i][1])

n = 100

sumatorio_beneficios = 0
sumatorio_número_beneficios = 1
lista_colusion = []

for i in range(empresas):
    lista_colusion.append([])

    plt.plot(lista_numero_episodios, lista_beneficios[i])
    sumatorio_número_beneficios = sumatorio_número_beneficios + len(lista_beneficios[i])

    for j in range(len(lista_beneficios[i])):

        sumatorio_beneficios = sumatorio_beneficios + lista_beneficios[i][j]
        
        beneficios_promedio = sumatorio_beneficios/sumatorio_número_beneficios
        lista_colusion[i].append(beneficios_promedio)

lista_colusion_definitiva = []
lista_colusion_predef = []

for j in range(len(lista_beneficios[0])):
    promedio_final = 0
    for i in range(empresas):
        promedio_final = promedio_final + lista_colusion[i][j]
        if i == (empresas-1):
            lista_colusion_predef.append(promedio_final)
    colusion_final = ((lista_colusion_predef[j]) - (beneficio_nash)) / ((beneficio_maximo) - (beneficio_nash))
    lista_colusion_definitiva.append(colusion_final)

plt.axhline(y=beneficios_promedio, color='blue', linestyle='--', label = 'Límite colusorio')
plt.axvline(x=198000, color='red', linestyle='--', label = 'Greedy')
plt.xlabel('Episodios')
plt.ylabel('Beneficios')
plt.show()

plt.plot(lista_numero_episodios, lista_colusion_definitiva)
plt.axhline(y=0, color='blue', linestyle='--', label = 'Competencia')
plt.axhline(y=1, color='red', linestyle='--', label = 'Colusion')
plt.show()

xmin = 0
ymin = 0
promedios_moviles = []

for j in range(empresas):
    promedios_moviles.append([])

for i in range(n-1, len(lista_precios[0])):
    for j in range(empresas):
        promedio = sum(lista_precios[j][i-n+1:i+1])/n
        promedios_moviles[j].append(promedio)

delete = n - 1
for sublista in lista_precios:
    sublista[:] = sublista[:-delete]

lista_numero_episodios = lista_numero_episodios[:-delete]

for i in range(len(lista_precios)):
    plt.plot(lista_numero_episodios, lista_precios[i], label='Evolución de precios empresa {}'.format(i+1))

for i in range(len(promedios_moviles)):
    plt.plot(lista_numero_episodios, promedios_moviles[i], label='Movimientos promedio empresa {}'.format(i+1))

plt.axhline(y=precio_minimo, color='blue', linestyle='--', label = 'Equilibrio de Nash')
plt.axhline(y=precio_beneficio_maximo, color='red', linestyle='--', label = 'Beneficio monopolístico')
plt.axvline(x=198000, color='black', linestyle='--', label = 'Greedy')

plt.xlabel('Episodios')
plt.ylabel('Precio')

for i in range(len(lista_episodios_convergentes)-n):
    if lista_episodios_convergentes[i]==1 and lista_episodios_convergentes[i+1]==1 and i!=(len(lista_episodios_convergentes)-n):
        plt.axvspan(i, i+1, alpha=0.3, color='y')

plt.xlim(left=xmin)
plt.ylim(bottom=ymin)
plt.show()

for i in range(len(lista_precios)):
    plt.plot(lista_numero_episodios, lista_precios[i], alpha = 0.5, label='Evolución de precios empresa {}'.format(i+1))

for i in range(len(promedios_moviles)):
    plt.plot(lista_numero_episodios, promedios_moviles[i], alpha = 0.5, label='Movimientos promedio empresa {}'.format(i+1))

plt.axhline(y=precio_minimo, color='blue', linestyle='--', label = 'Equilibrio de Nash')
plt.axhline(y=precio_beneficio_maximo, color='red', linestyle='--', label = 'Beneficio monopolístico')
plt.axvline(x=198000, color='black', linestyle='--', label = 'Greedy')

plt.xlabel('Episodios')
plt.ylabel('Precio')

for i in range(len(lista_episodios_convergentes)-n):
    if lista_episodios_convergentes[i]==1 and lista_episodios_convergentes[i+1]==1 and i!=(len(lista_episodios_convergentes)-n):
        plt.axvspan(i, i+1, alpha=0.3, color='y')

plt.xlim(left=xmin)
plt.ylim(bottom=ymin)

plt.show()

end2 = time()
print(f"Se han tardado {end2-start2} segundos en completar la simulación de Bertrand")