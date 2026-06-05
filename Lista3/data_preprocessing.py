import logging
from pathlib import Path
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

def load_dataset(verbose = False) -> DataFrame:
    file_path = Path(__file__).resolve().parents[1] / "resources" / "sleep_health_dataset.csv"
    df = pd.read_csv(file_path)

    if 'person_id' in df.columns:
        df = df.drop('person_id', axis=1)

    logger.debug(f"\nRozmiar calego zbioru danych: {df.shape[0]} wierszy x {df.shape[1]} kolumn\n")
    logger.debug(df.info())
    # logger.debug(df.describe().round(2))

    return df


def extract_target(df: DataFrame, target_col: str) -> tuple[DataFrame, Series]:
    y = df[target_col]
    X = df.drop(columns=[target_col])

    logger.debug(f"\nZmienne niezalezne (cechy): {X.columns.tolist()}")
    logger.debug(f"\nZmienna zalezna (target): {target_col}")
    
    return X, y

def df_label_encoding(df: DataFrame) -> DataFrame:
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    logger.debug(f"\nZmienne kategoryczne: {categorical_cols}")
    logger.debug(f"Zmienne numeryczne: {numerical_cols}")

    label_encoders = {}
    df_encoded = df.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        logger.debug(f"\nKodowanie {col}:")
        for i, class_name in enumerate(le.classes_):
            logger.debug(f"  {class_name} -> {i}")

    logger.debug(df_encoded.head(5))

    return df_encoded