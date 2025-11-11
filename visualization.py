import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 

class VisualizeDataset:
    def __init__(self, data):
        """
        Classe per visualizzare ed esplorare un dataset Pandas.
        """
        self.data = data
        sns.set(style="whitegrid", palette="muted", font_scale=1.1)


    #NUOVO
    def overview(self, n_rows=10, show_missing=True ):
        """Mostra info generali sul dataset."""
        print("Shape:", self.data.shape)
        print("\nTipi di dato:\n", self.data.dtypes.value_counts())
        print("\nPrime righe:")
        self.data.head()

        if show_missing:
            missing = self.data.isna().sum()
            print("\n Valori mancanti per colonna:")
            print(missing[missing > 0] if missing.any() else "Nessun valore mancante.")

        print("\nStatistiche descrittive:")
        numeriche = self.data.select_dtypes(include=['number'])
        categoriche = self.data.select_dtypes(include=['object', 'category'])

        if not numeriche.empty:
            print("\n -- Numeriche: --")
            print(numeriche.describe(percentiles=[0.25, 0.5, 0.75]).to_string())

        if not categoriche.empty:
            print("\n -- Categoriche: --")
            print(categoriche.describe().to_string()) 


        
    
        

    #visualizzare distribuzione ogni colonna del dataframe (NUOVO)
    def distribuitions(self, max_unique=20, bins=30, top_n_categorie=20, normality_test=True):
        for col in self.data.columns:
            unique_values = self.data[col].nunique()
            print(f"\n Colonna: {col}")
            print(self.data[col].describe(include='all'))
            print("-" * 40)

            plt.figure(figsize=(12, 5))

    
            if pd.api.types.is_numeric_dtype(self.data[col]):
                plt.subplot(1, 3 , 1)
                sns.histplot(self.data[col].dropna(), kde=True, bins=bins)
                plt.title(f"Distribuzione di {col}")

                plt.subplot(1, 3, 2)
                sns.boxplot(x=self.data[col])
                plt.title(f"Boxplot di {col}")

                plt.subplot(1,3,3)
                stats.probplot(self.data[col].dropna(), dist="norm", plot=plt)
                plt.title(f"Q-Q plot di {col}")

                plt.tight_layout()
                plt.show()

                if normality_test:
                    stat, p = stats.shapiro(self.data[col].dropna())
                    print(f"Test di Shapiro-Wilk per normalità: stat={stat:.3f}, p={p:.3f}")
                    if p > 0.05:
                        print("La distribuzione sembra normale (non si rifiuta H0)")
                    else:
                        print("La distribuzione non è normale (si rifiuta H0)")

            else:
                unique_values = self.data[col].nunique()
                if unique_values <= max_unique:
                    sns.countplot(data=self.data, x=col, order=self.data[col].value_counts().index)
                    plt.title(f"Distribuzione di {col}")
                    plt.xlabel(col)
                    plt.ylabel("Conteggio")
                    plt.xticks(rotation=45)

                else:
                    print(f"Troppe categorie ({unique_values}) in {col}, mostra solo le prime {top_n_categorie}")
                    top_n_categorie = self.data[col].value_counts().nlargest(top_n_categorie).index
                    sns.countplot(data=self.data, x=col, order=top_n_categorie)
                    plt.title(f"Distribuzione di {col} (top {top_n_categorie})")
                    plt.xlabel(col)
                    plt.ylabel("Conteggio")    
                    plt.xticks(rotation=45)

                plt.tight_layout()
                plt.show()



    def missing_values(self, top_n=20):
        """Grafico delle colonne con più valori mancanti."""
        missing = self.data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(top_n)
        if len(missing) == 0:
            print("Nessun valore mancante nel dataset.")
            return

        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing.values, y=missing.index, color='tomato')
        plt.title(f"Top {top_n} colonne con valori mancanti")
        plt.xlabel("Numero di NaN")
        plt.ylabel("Colonna")
        plt.show()

    def correlation_heatmap(self, top_n=20):
        """Mostra una heatmap delle correlazioni tra le variabili numeriche più importanti."""
        numeric_df = self.data.select_dtypes(include=['number'])
        if numeric_df.empty:
            print("Nessuna colonna numerica trovata.")
            return

        corr = numeric_df.corr()
        # Se esiste la colonna target 'Weekly_Sales', scegli le colonne con le
        # correlazioni assolute maggiori rispetto a Weekly_Sales.
        # Altrimenti, prendi semplicemente le prime top_n colonne disponibili.
        if 'Weekly_Sales' in corr.columns:
            top_corr = corr['Weekly_Sales'].abs().nlargest(top_n).index
        else:
            top_corr = corr.columns[:top_n]
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr.loc[top_corr, top_corr], annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Heatmap di correlazione (top {top_n})")
        plt.show()


    def jointplot_feature(self, feature, target="Weekly_Sales", kind="reg"):
        """Jointplot (scatter + distribuzioni) tra feature e target."""
        if feature not in self.data.columns or target not in self.data.columns:
            print("Feature o target non trovati.")
            return
        sns.jointplot(data=self.data, x=feature, y=target, kind=kind, height=7, marginal_kws=dict(bins=30))
        plt.suptitle(f"Jointplot: {feature} vs {target}", y=1.02)
        plt.show()




# Esempio (usando il dataset House Prices)
df = pd.read_csv("dataset.csv")

viz = VisualizeDataset(df)
#viz.overview()
viz.missing_values()
#viz.jointplot_feature("Size", "Weekly_Sales")

#viz.distribuitions()
viz.correlation_heatmap()
#viz.distribuitions()

corr = data.corr()
f, ax = plt.subplots(figsize=(15, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

