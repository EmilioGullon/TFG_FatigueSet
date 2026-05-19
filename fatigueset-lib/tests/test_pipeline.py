from typing import Any, Dict, cast

import pandas as pd

from fatigueset.pipeline import FatigueSetPipeline


def _write_csv(path, content):
    path.write_text(content, encoding='utf-8')


def test_pipeline_end_to_end_with_minimal_dataset(tmp_path):
    base = tmp_path / 'fatigueset'
    participant = base / '01' / '01'
    participant.mkdir(parents=True)

    _write_csv(
        base / 'metadata.csv',
        'participant_id,low_session,medium_session,high_session\n1,1,2,3\n',
    )
    _write_csv(
        participant / 'exp_fatigue.csv',
        'measurementNumber,physicalFatigueScore,mentalFatigueScore\n0,10,20\n1,15,30\n2,25,45\n',
    )
    _write_csv(
        participant / 'chest_physiology_summary.csv',
        'timestamp,hr,br,hrv\n1,60,12,40\n2,61,13,41\n3,62,14,42\n',
    )
    _write_csv(
        participant / 'wrist_eda.csv',
        'eda\n0.1\n0.2\n0.3\n',
    )
    _write_csv(
        participant / 'exp_nback.csv',
        'measurementNumber,isCorrectResponse,responseTime\n0,1,500\n0,0,650\n1,1,520\n',
    )
    _write_csv(
        participant / 'exp_crt.csv',
        'measurementNumber,isCorrectResponse,responseTime\n0,1,300\n1,1,320\n2,0,350\n',
    )

    pipeline = FatigueSetPipeline(dataset_path=str(base), participantes=['01'], sesiones=['01'])

    resultados = cast(Dict[str, Any], pipeline.ejecutar(verbose=False, incluir_ventanas=False, normalizar=True))

    assert 'dataset' in resultados
    assert resultados['dataset']['fatiga'] is not None
    assert not resultados['validacion'].empty
    assert not resultados['fatigabilidad'].empty
    assert not resultados['ml'].empty
    assert 'fatiga_mental' in resultados['ml'].columns
    assert not resultados['ml_normalizado'].empty
    assert 'fatiga_mental' in resultados['correlaciones']

    df_ventanas = pipeline.crear_ventanas(resultados['ml'], window_size=2, step=1, group_cols=None)
    assert isinstance(df_ventanas, pd.DataFrame)
    assert not df_ventanas.empty