import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, normal_ad
from scipy.stats import jarque_bera
from scipy.stats import norm
from scipy.stats import chi2
import pandas as pd
import numpy as np
import seaborn as sns

class TimeSeriesAnalysis:
    def __init__(self, lags: int = 30, alpha = 0.05):
        """
        Constructor de la clase. 
        
        Parámetros
        ---
        lags: int
            Número de lags a considerar para el ACF y PACF. Por defecto 30.
        alpha: float
            Nivel de significancia para las pruebas estadísticas. Por defecto 0.05. Se puede cambiar en cada método.
        """

        self.lags = lags
        self.settings = {
            'alpha': alpha, # Nivel de significancia para las pruebas estadísticas
            'adjusted': True, # Corrige sesgo muestral al estimar autocorrelaciones
            'pacf_method': 'ywm', # Método de Yule-Walker para el PACF
            'missing': 'drop', # Elimina los valores NaN de la serie temporal
            'bartlett_confint': True, # Intervalo de confianza de Bartlett (VER)
            'fft': None, # Si la serie tiene más de 10.000 observaciones, se usa FFT para el ACF

        }

    def plot_acf_pacf(self, serie, variable_title = None):
        """
        Plotea el ACF y PACF de la serie temporal para análisis gráfico del modelo.

        Parámetros
        ---
        alpha: float
            Nivel de significancia para las pruebas estadísticas. Por defecto 0.05. Se puede cambiar en cada método.
        variable_title: str
            Titulo de la variable a analizar. Por defecto None.

        """
        fig, axes = plt.subplots(1, 2, figsize = (12, 4))

        # Si el self.settings['fft'] no tiene nada, se añade:
        if self.settings['fft'] is None:
            self.settings['fft'] = len(serie.dropna()) > 10_000
            if self.settings['fft']:
                print("La serie tiene más de 10.000 observaciones. Se usará FFT para el ACF.")

        plot_acf(serie, 
                 ax = axes[0], 
                 lags = self.lags, 
                 alpha = self.settings['alpha'], 
                 adjusted = self.settings['adjusted'],
                 fft = self.settings['fft'],
                 missing = self.settings['missing'],
                 title = 'ACF', # Titulo del grafico
                 bartlett_confint = self.settings['bartlett_confint'], 
                 ) 
        
        plot_pacf(serie, ax=axes[1], 
                  lags = self.lags, 
                  alpha = self.settings['alpha'], 
                  method = self.settings['pacf_method'],
                  title = 'PACF',
                  )
        
        if variable_title is not None:
            fig.suptitle(f"ACF y PACF para: {variable_title}", fontsize=14)
        else:
            pass

        plt.tight_layout(rect = [0, 0, 1, 0.95])  # Ajusta para que no se superponga el título
        plt.show()

    # def suggest_model_order(self, serie, threshold: float = None):
    #     """
    #     Sugiere el orden del modelo ARIMA a partir del análisis de ACF y PACF.
    #     Se basa en el criterio visual clásico: si la PACF corta antes que la ACF → modelo AR(p), 
    #     si la ACF corta antes → modelo MA(q), y si ambas cortan parecido → modelo ARMA(p,q).

    #     La función normaliza la serie antes del análisis para garantizar estabilidad en la 
    #     estimación de correlaciones y hacer que el threshold tenga sentido estadístico.

    #     Parámetros
    #     ----------
    #     threshold : float, opcional
    #         Umbral para considerar que la ACF o PACF "corta" (valor absoluto menor al umbral).
    #         Si es None (por defecto), se calcula automáticamente como 1.96 / sqrt(T),
    #         siendo T el número de observaciones efectivas (como en los intervalos de confianza 
    #         bajo H₀: ruido blanco).

    #     Devuelve
    #     --------
    #     (int, int)
    #         Ordenes sugeridos para el componente AR y MA respectivamente.
    #     """

    #     # Paso 1: Limpiamos y normalizamos la serie (VER)
    #     serie_std = (serie - serie.mean()) / serie.std()

    #     # Paso 2: Se calcula el threshold si no se pasa como argumento y se define fft
    #     if threshold is None:
    #         z_crit = norm.ppf(1 - self.settings['alpha'] / 2)
    #         threshold = z_crit / np.sqrt(len(serie_std))

    #     if self.settings['fft'] is None:
    #         self.settings['fft'] = len(serie.dropna()) > 10_000
    #         if self.settings['fft']:
    #             print("La serie tiene más de 10.000 observaciones. Se usará FFT para el ACF.")

    #     # Paso 3: calculamos ACF y PACF normalizados
    #     acf_vals, _ = acf(serie_std, 
    #                     nlags = self.lags, 
    #                     fft = self.settings['fft'], 
    #                     adjusted = self.settings['adjusted'], 
    #                     missing = self.settings['missing'], 
    #                     alpha = self.settings['alpha'],
    #                     bartlett_confint = self.settings['bartlett_confint'],
    #                    )
        
    #     pacf_vals, _ = pacf(serie_std, 
    #                      nlags = self.lags,
    #                      method = self.settings['pacf_method'],
    #                      alpha = self.settings['alpha'],
    #                      ) 

    #     # Paso 4: identificar orden del corte
    #     ar_order = np.argmax(np.abs(pacf_vals) < threshold)
    #     ma_order = np.argmax(np.abs(acf_vals) < threshold)

    #     # Paso 5: imprimir sugerencia
    #     print("\nSugerencia automática (tentativa):")
    #     if ar_order < ma_order:
    #         print(f"PACF corta antes: posible modelo AR({ar_order})")
    #         print(f"Sugerencia por ACF (MA): {ma_order}")
    #     elif ma_order < ar_order:
    #         print(f"ACF corta antes: posible modelo MA({ma_order})")
    #         print(f"Sugerencia por PACF (AR): {ar_order}")
    #     else:
    #         print(f"Ambas cortan parecido, posible ARMA({ar_order},{ma_order})")

    #     return ar_order, ma_order
    
    # def suggest_model_order(self, serie, max_order: int = 5, tipo: str = "ARMA", umbral_llf: float = 5.0):
    #     """
    #     Sugiere el orden del modelo ARIMA usando un criterio de mejora en la log-verosimilitud (LLF).
    #     Empieza con el modelo más chico y va aumentando el orden hasta que la mejora en LLF es marginal.

    #     Parámetros
    #     ----------
    #     max_order : int
    #         Orden máximo a considerar para p y q.
    #     tipo : str
    #         Puede ser "AR", "MA", o "ARMA". Define la estructura a testear.
    #     umbral_llf : float
    #         Umbral mínimo de mejora en log-verosimilitud para justificar aumento de orden.

    #     Retorna
    #     -------
    #     Tuple[int, int]
    #         Orden sugerido (p, q) según criterio de mejora marginal.
    #     """

    #     llf_anterior = None
    #     orden_sugerido = (0, 0)

    #     for i in range(1, max_order + 1):
    #         if tipo == "AR":
    #             orden = (i, 0, 0)
    #         elif tipo == "MA":
    #             orden = (0, 0, i)
    #         else:  # ARMA
    #             orden = (i, 0, i)

    #         try:
    #             modelo = ARIMA(serie, order=orden).fit()
    #             llf_actual = modelo.llf

    #             if llf_anterior is not None:
    #                 mejora = llf_actual - llf_anterior
    #                 if mejora < umbral_llf:
    #                     break

    #             llf_anterior = llf_actual
    #             orden_sugerido = orden

    #         except Exception as e:
    #             print(f"Fallo el modelo {orden}: {e}")
    #             continue

    #     print(f"\nOrden sugerido según LLF marginal: AR({orden_sugerido[0]}), MA({orden_sugerido[2]}) y estacionariedad d={orden_sugerido[1]}")
    #     return orden_sugerido[0], orden_sugerido[2]

    def suggest_model_order(self, serie, umbral_llf: float=None, max_order: int = 5, tipo: str = "ARMA",  threshold: float = 0.05):
        """
        Sugiere el orden del modelo ARIMA según dos criterios independientes:
        
        1) Mejora secuencial de log-verosimilitud (LLF): 
        aumenta el orden hasta que la mejora marginal en LLF sea menor al umbral.
        
        2) Significancia estadística de coeficientes (p-values < threshold):
        sugiere el mayor orden tal que todos los coeficientes relevantes sean significativos.

        Parámetros
        ----------
        serie : array-like
            Serie temporal a modelar.
        max_order : int
            Orden máximo a considerar para p y q.
        tipo : str
            "AR", "MA", o "ARMA". Define la estructura a testear.
        umbral_llf : float
            Umbral mínimo de mejora en log-verosimilitud para justificar aumento de orden.
        threshold : float
            Umbral de significancia para los p-values de los coeficientes.

        Retorna
        -------
        dict
            {"llf": (p, q), "pval": (p, q)}
        """
        serie = serie.dropna()

        if umbral_llf is None:
            umbral_llf = chi2.ppf(1 - self.settings['alpha'], df=1) / 2

        llf_anterior = None
        orden_llf = (0, 0, 0)
        orden_pval = (0, 0, 0)

        for i in range(1, max_order + 1):
            if tipo == "AR":
                orden = (i, 0, 0)
                coeficientes = [f'ar.L{j}' for j in range(1, i+1)]
            elif tipo == "MA":
                orden = (0, 0, i)
                coeficientes = [f'ma.L{j}' for j in range(1, i+1)]
            else:  # ARMA
                orden = (i, 0, i)
                coeficientes = [f'ar.L{j}' for j in range(1, i+1)] + [f'ma.L{j}' for j in range(1, i+1)]

            try:
                modelo = ARIMA(serie, order=orden).fit()
                llf_actual = modelo.llf
                pvalues = modelo.pvalues

                # Actualizar sugerencia por p-value
                if all(pvalues.get(c, 1) < threshold for c in coeficientes):
                    orden_pval = orden

                # Actualizar sugerencia por log-verosimilitud
                if llf_anterior is not None:
                    mejora = llf_actual - llf_anterior
                    if mejora < umbral_llf:
                        break

                llf_anterior = llf_actual
                orden_llf = orden

            except Exception as e:
                print(f"Fallo el modelo {orden}: {e}")
                continue

            print("\nSugerencias:")

            if orden_llf[0] > 0 or orden_llf[2] > 0:
                print(f" Por log-verosimilitud marginal: ARMA({orden_llf[0]}, {orden_llf[2]})" if orden_llf[0] > 0 and orden_llf[2] > 0
                    else f" AR({orden_llf[0]})" if orden_llf[0] > 0
                    else f" MA({orden_llf[2]})")

            if orden_pval[0] > 0 or orden_pval[2] > 0:
                print(f" Por significancia estadística: ARMA({orden_pval[0]}, {orden_pval[2]})" if orden_pval[0] > 0 and orden_pval[2] > 0
                    else f" AR({orden_pval[0]})" if orden_pval[0] > 0
                    else f" MA({orden_pval[2]})")

            return {"llf": (orden_llf[0], orden_llf[2]), "pval": (orden_pval[0], orden_pval[2])}


    def correlograma_tabla(self, serie, latex=False, period=None):
        """
        Devuelve tabla con ACF, PACF, Q-Stat y p-valores al estilo EViews.
        Si latex=True, devuelve el string en formato LaTeX.

        Parametros
        ---
        latex: bool
            Si True, devuelve el string en formato LaTeX.
        period: int
            Periodo de la serie temporal. Si es estacional, se puede poner el valor.
            Si no se pone, se usa el valor por defecto (None).
        """
        # Si el self.settings['fft'] no tiene nada, se añade:
        if self.settings['fft'] is None:
            self.settings['fft'] = len(serie.dropna()) > 10_000
            if self.settings['fft']:
                print("La serie tiene más de 10.000 observaciones. Se usará FFT para el ACF.")

        acf_vals, _ = acf(serie, 
                        nlags = self.lags, 
                        fft = self.settings['fft'], 
                        adjusted = self.settings['adjusted'], 
                        missing = self.settings['missing'], 
                        alpha = self.settings['alpha'],
                        bartlett_confint = self.settings['bartlett_confint'],
                       )
        pacf_vals, _ = pacf(serie, 
                         nlags = self.lags,
                         method = self.settings['pacf_method'],
                         alpha = self.settings['alpha'],
                         ) 
        
        ljungbox = acorr_ljungbox(serie, lags=self.lags, 
                                  return_df=True, 
                                  period=period,
                                boxpierce=True,
                                )

        tabla = pd.DataFrame({
            "Lag": np.arange(self.lags + 1),
            "AC": np.round(acf_vals, 3),
            "PAC": np.round(pacf_vals, 3),
            "Q-Stat": np.insert(np.round(ljungbox["lb_stat"].values, 3), 0, np.nan),
            "Prob": np.insert(np.round(ljungbox["lb_pvalue"].values, 3), 0, np.nan)
        })

        if latex:
            tabla = tabla.to_latex(index=False, float_format="%.3f", escape=False, column_format="lcccc")
            tabla = tabla.replace("\\\\", "\\\\ \\hline")
            tabla = tabla.replace("\\toprule", "\\hline \\hline")
            tabla = tabla.replace("\\bottomrule", "\\hline")
            tabla = tabla.replace("\\midrule", "\\hline")

        return tabla
    
    def test_ljung_box(self, serie, period=None):
        """
        Realiza la prueba de Ljung-Box para verificar la independencia de los residuos. 
        Si el p-value es muy chico se rechaza la hipótesis nula de independencia, con lo
        cual el modelo no capturó toda la información de la serie temporal.

        Parametros
        ---
        period: int
            Periodo de la serie temporal. Si es estacional, se puede poner el valor.
            Si no se pone, se usa el valor por defecto (None).
            Si la serie es estacional, se puede poner el valor del periodo estacional.
        """

        ljungbox = acorr_ljungbox(serie, lags=self.lags, 
                                  return_df=True, 
                                  period=period,
                                  boxpierce=True,
                                  )
        
        return ljungbox

    def test_normalidad_residuos(self, resid, alpha=None):
        """
        Realiza la prueba de normalidad de los residuos utilizando el test Jarque-Bera.

        Parámetros
        ----------
        resid : array-like
            Residuos del modelo a testear. Obligatorio.
        alpha : float
            Nivel de significancia. Si None, se toma de self.settings['alpha'].

        Returns
        -------
        dict con estadístico y p-valor
        """
        if resid is None:
            raise ValueError("Debes pasar un vector de residuos (resid) para este test.")
        if alpha is None:
            alpha = self.settings['alpha']

        jb_stat, jb_pval = jarque_bera(resid)

        print("\nTest de normalidad Jarque-Bera sobre residuos:")
        print(f"Estadístico JB: {jb_stat:.3f} - p-valor: {jb_pval:.3f}")
        if jb_pval < alpha:
            print(f"→ Se rechaza la normalidad a nivel {alpha*100:.1f}%.")
        else:
            print(f"→ No se rechaza la normalidad a nivel {alpha*100:.1f}%.")

        return {"jb_stat": jb_stat, "jb_p_value": jb_pval}

    def fit_arima(self,
                endog:pd.DataFrame,
                exog=None,
                order=(0, 0, 0),
                seasonal_order=(0, 0, 0, 0),
                trend=None,
                enforce_stationarity=True,
                enforce_invertibility=True,
                concentrate_scale=False,
                trend_offset=1,
                dates=None,
                freq=None,
                missing=None,
                validate_specification=True,
                residuals_plot=False, 
                sugerencia_tipo = "ARMA",
                ):
        """
        Ajusta un modelo ARIMA a la serie temporal y muestra resumen y diagnóstico.

        Parámetros
        ----------
        endog : pd.DataFrame
            Serie temporal endógena a ajustar. Debe ser un objeto de pandas.
        exog : array-like, opcional
            Variables exógenas si se desea incluir regresores adicionales.
        order : tuple
            Orden (p, d, q) del modelo ARIMA. p es el orden autoregresivo, d es el grado de diferenciación
            para hacer la serie estacionaria (d=0, la serie ya es estacionaria, d=1 hacemos primera diferencia,
            etc.), y q es el orden del MA process.
        seasonal_order : tuple
            Orden estacional del modelo SARIMA en la forma (P, D, Q, s), donde:
            - P: orden autorregresivo estacional
            - D: orden de diferenciación estacional
            - Q: orden de media móvil estacional
            - s: periodicidad de la estacionalidad (por ejemplo, 12 para datos mensuales con ciclo anual)

            El modelo resultante queda expresado como:

                Φ_P(L^s) · (1 - L^s)^D · (1 - L)^d · y_t = Θ_Q(L^s) · Θ_q(L) · ε_t

            Donde **L** es el operador rezago (L·y_t = y_{t-1}).
            Por defecto, es (0, 0, 0, 0), lo que significa que no se incluye un componente estacional.
        trend : {'n', 'c', 't', 'ct'} o None
            Especifica si se incluye una constante y/o tendencia lineal en el modelo ARIMA.
            - 'n': sin constante ni tendencia
            - 'c': constante (intercepto)
            - 't': tendencia lineal
            - 'ct': constante + tendencia lineal
            - None: el modelo decide automáticamente en base al orden de diferenciación

            La tendencia se incorpora **antes** de la diferenciación. Por ejemplo, con trend='c':

                y_t = μ + ARIMA(p,d,q) + ε_t
        enforce_stationarity : bool
            Si True, impone estacionariedad en la estimación, es decir, no permite que el componente
            AR tiene raíces fuera del círculo unitario. Esto es útil para evitar problemas de
            inestabilidad en el modelo. Si False, permite raíces fuera del círculo unitario.
        enforce_invertibility : bool, default=True
            Si True, el modelo impone que los coeficientes MA generen un proceso invertible,
            es decir, que las raíces del polinomio MA estén fuera del círculo unitario.

            Esto permite una representación equivalente AR(∞) estable y evita soluciones 
            numéricamente inestables. Si False, se permite estimar modelos no invertibles, 
            útil en fases exploratorias o cuando se desea comparar especificaciones.
        concentrate_scale : bool, default=False
            Si True, la varianza del error (\sigma^2) se concentra fuera del proceso 
            de optimización, lo que reduce el número de parámetros estimados y acelera 
            el proceso.

            Sin embargo, al concentrar la escala:
            - No se estima explícitamente \sigma^2
            - No se obtiene su intervalo de confianza
            - La matriz de covarianza de parámetros no la incluye

            Si se desea obtener inferencia completa sobre la varianza, dejar en False.
        trend_offset : int, default=1
            Define desde qué valor numérico comienza el regressor de tendencia lineal cuando se incluye 
            `trend='t'` o `trend='ct'` en el modelo.

            Esto no modifica los datos ni impone discontinuidades: simplemente cambia el punto desde 
            el cual se empieza a contar la variable temporal. Por ejemplo:
            - trend_offset=0 → tendencia = [0, 1, 2, 3, ...]
            - trend_offset=5 → tendencia = [5, 6, 7, 8, ...]

            El valor de `trend_offset` afecta la interpretación del intercepto (μ), pero no cambia 
            la pendiente (β) de la tendencia. No implica un cambio estructural ni activa la tendencia 
            a partir de cierto período.
        dates : array-like, opcional
            Fechas explícitas para la serie.
        freq : str o pd.DateOffset
            Frecuencia temporal.
        missing : str
            Método para manejar valores faltantes: 'none', 'drop', etc.
        validate_specification : bool, default=True
            Si True, el modelo valida automáticamente que la especificación sea coherente. 
            Verifica consistencia entre los rezagos AR/MA y el largo de la serie, los términos 
            de tendencia, las diferenciaciones y las variables exógenas. Recomendado para asegurar que el 
            modelo tiene sentido estadístico y evitar errores ocultos.

        residuals_plot : bool
            Si True, muestra histograma y ACF de los residuos.

        Retorna
        -------
        dict con el modelo, AIC, BIC y resultado del test de Ljung-Box.
        """
        if missing is None:
            missing = self.settings['missing']

        model = ARIMA(
            endog=endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
            concentrate_scale=concentrate_scale,
            trend_offset=trend_offset,
            dates=dates,
            freq=freq,
            missing=missing,
            validate_specification=validate_specification
        )

        # Ajuste del modelo
        result = model.fit()

        # Resumen del modelo
        print("\nResumen del modelo ARIMA:\n")
        print(result.summary())

        # Test de Ljung-Box sobre residuos
        resid = result.resid
        ljungbox_results = self.test_ljung_box(resid)
        # print("\nResultados de la prueba de Ljung-Box sobre residuos:\n")
        # print(ljungbox_results)

        # Interpretación y normalidad:
        self.suggest_model_order(endog, tipo= sugerencia_tipo)
 
        if residuals_plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(resid, ax=axes[0], kde=True)
            axes[0].set_title(f"Histograma de residuos - ARIMA{order}")
            plot_acf(resid, ax=axes[1], lags=self.lags)
            axes[1].set_title("ACF de residuos")
            plt.tight_layout()
            plt.show()

        return {
                "modelo": result,
                "aic": result.aic,
                "bic": result.bic,
                "ljungbox": ljungbox_results
            }
    
    ## Sumar estimación VAR, permitir elegir variables exogenas y endogenas. Permitir fittear VAR, VECM y bayesiano.
    ### Mostrar t-stat, coefiicente y standar errors

    ## Armar VAR order selection method: le das un numero maximo de lags, te regresa contra todos, calcula 
    ## los criterios de Akaike, Schwarz y Hannan-Quinn, LR y likelihood. Te marca cual es mejor para cada criterio. 

    ## Al estimar un VAR, deberias poder restringir que algunos coeficientes sean cero
    ## Sumar test de exclusion VECM 
    ## Algun menu de lag structure deberia agregar el lag exclusion test. Ver PC Gets para elegir la estructura. 