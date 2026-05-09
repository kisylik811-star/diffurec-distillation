"""
Baseline numbers cited verbatim from DiffuRec, Table 2.

These are not re-computed — they are used as reference rows in the main
results table (with proper attribution in the dissertation text).

Source:
  Li, Z., Li, C., Sun, A. "DiffuRec: A Diffusion Model for Sequential
  Recommendation". ACM TOIS, 2023. Table 2 (page 16 in the arXiv preprint).
  Values are percentages (HR/NDCG × 100).

Usage:
  from literature_baselines import LITERATURE_BASELINES
  LITERATURE_BASELINES['amazon_beauty']['SASRec']['HR@10']  # -> 6.2648
"""

LITERATURE_BASELINES = {
    'amazon_beauty': {
        'GRU4Rec': {
            'HR@5': 1.0112, 'HR@10': 1.9370, 'HR@20': 3.8531,
            'NDCG@5': 0.6084, 'NDCG@10': 0.9029, 'NDCG@20': 1.3804,
        },
        'Caser': {
            'HR@5': 1.6188, 'HR@10': 2.8166, 'HR@20': 4.4048,
            'NDCG@5': 0.9758, 'NDCG@10': 1.3602, 'NDCG@20': 1.7595,
        },
        'SASRec': {
            'HR@5': 3.2688, 'HR@10': 6.2648, 'HR@20': 8.9791,
            'NDCG@5': 2.3989, 'NDCG@10': 3.2305, 'NDCG@20': 3.6563,
        },
        'BERT4Rec': {
            'HR@5': 2.1326, 'HR@10': 3.7160, 'HR@20': 5.7922,
            'NDCG@5': 1.3207, 'NDCG@10': 1.8291, 'NDCG@20': 2.3541,
        },
        'ComiRec': {
            'HR@5': 2.0495, 'HR@10': 4.4545, 'HR@20': 7.6968,
            'NDCG@5': 1.0503, 'NDCG@10': 1.8306, 'NDCG@20': 2.6451,
        },
        'TiMiRec': {
            'HR@5': 1.9044, 'HR@10': 3.3434, 'HR@20': 5.1674,
            'NDCG@5': 1.2438, 'NDCG@10': 1.7044, 'NDCG@20': 2.1627,
        },
        'SVAE': {
            'HR@5': 0.9943, 'HR@10': 1.9745, 'HR@20': 3.1552,
            'NDCG@5': 0.6702, 'NDCG@10': 0.9863, 'NDCG@20': 1.2867,
        },
        'ACVAE': {
            'HR@5': 2.4672, 'HR@10': 3.8832, 'HR@20': 6.1224,
            'NDCG@5': 1.6858, 'NDCG@10': 2.1389, 'NDCG@20': 2.7020,
        },
        'STOSA': {
            'HR@5': 3.5457, 'HR@10': 6.2048, 'HR@20': 9.5939,
            'NDCG@5': 2.5554, 'NDCG@10': 3.2085, 'NDCG@20': 3.7609,
        },
        'DiffuRec_paper': {
            'HR@5': 5.5758, 'HR@10': 7.9068, 'HR@20': 11.1098,
            'NDCG@5': 4.0047, 'NDCG@10': 4.7494, 'NDCG@20': 5.5566,
        },
    },

    'toys': {
        'GRU4Rec': {
            'HR@5': 1.1009, 'HR@10': 1.8553, 'HR@20': 3.1827,
            'NDCG@5': 0.6983, 'NDCG@10': 0.9396, 'NDCG@20': 1.2724,
        },
        'Caser': {
            'HR@5': 0.9622, 'HR@10': 1.8317, 'HR@20': 2.9500,
            'NDCG@5': 0.5707, 'NDCG@10': 0.8510, 'NDCG@20': 1.1293,
        },
        'SASRec': {
            'HR@5': 4.5333, 'HR@10': 6.5496, 'HR@20': 9.2263,
            'NDCG@5': 3.0105, 'NDCG@10': 3.7533, 'NDCG@20': 4.3323,
        },
        'BERT4Rec': {
            'HR@5': 1.9260, 'HR@10': 2.9312, 'HR@20': 4.5889,
            'NDCG@5': 1.1630, 'NDCG@10': 1.4870, 'NDCG@20': 1.9038,
        },
        'ComiRec': {
            'HR@5': 2.3026, 'HR@10': 4.2901, 'HR@20': 6.9357,
            'NDCG@5': 1.1571, 'NDCG@10': 1.7953, 'NDCG@20': 2.4631,
        },
        'TiMiRec': {
            'HR@5': 1.1631, 'HR@10': 1.8169, 'HR@20': 2.7156,
            'NDCG@5': 0.7051, 'NDCG@10': 0.9123, 'NDCG@20': 1.1374,
        },
        'SVAE': {
            'HR@5': 0.9109, 'HR@10': 1.3683, 'HR@20': 1.9239,
            'NDCG@5': 0.5580, 'NDCG@10': 0.7063, 'NDCG@20': 0.8446,
        },
        'ACVAE': {
            'HR@5': 2.1897, 'HR@10': 3.0749, 'HR@20': 4.4061,
            'NDCG@5': 1.5604, 'NDCG@10': 1.8452, 'NDCG@20': 2.1814,
        },
        'STOSA': {
            'HR@5': 4.2236, 'HR@10': 6.9393, 'HR@20': 9.5096,
            'NDCG@5': 3.1017, 'NDCG@10': 3.8806, 'NDCG@20': 4.3789,
        },
        'DiffuRec_paper': {
            'HR@5': 5.5650, 'HR@10': 7.4587, 'HR@20': 9.8417,
            'NDCG@5': 4.1667, 'NDCG@10': 4.7724, 'NDCG@20': 5.3684,
        },
    },

    'ml-1m': {
        'GRU4Rec': {
            'HR@5': 5.1139, 'HR@10': 10.1664, 'HR@20': 18.6995,
            'NDCG@5': 3.0529, 'NDCG@10': 4.6754, 'NDCG@20': 6.8228,
        },
        'Caser': {
            'HR@5': 7.1401, 'HR@10': 13.3792, 'HR@20': 22.5507,
            'NDCG@5': 4.1550, 'NDCG@10': 6.1400, 'NDCG@20': 8.4304,
        },
        'SASRec': {
            'HR@5': 9.3812, 'HR@10': 16.8941, 'HR@20': 28.318,
            'NDCG@5': 5.3165, 'NDCG@10': 7.7277, 'NDCG@20': 10.5946,
        },
        'BERT4Rec': {
            'HR@5': 13.6393, 'HR@10': 20.5675, 'HR@20': 29.9479,
            'NDCG@5': 8.8922, 'NDCG@10': 11.1251, 'NDCG@20': 13.4763,
        },
        'ComiRec': {
            'HR@5': 6.1073, 'HR@10': 12.0406, 'HR@20': 21.0094,
            'NDCG@5': 3.5214, 'NDCG@10': 5.4076, 'NDCG@20': 7.6502,
        },
        'TiMiRec': {
            'HR@5': 16.2176, 'HR@10': 23.7142, 'HR@20': 33.2293,
            'NDCG@5': 10.8796, 'NDCG@10': 13.3059, 'NDCG@20': 15.7019,
        },
        'SVAE': {
            'HR@5': 1.4869, 'HR@10': 2.7189, 'HR@20': 5.0326,
            'NDCG@5': 0.9587, 'NDCG@10': 1.2302, 'NDCG@20': 1.8251,
        },
        'ACVAE': {
            'HR@5': 12.7167, 'HR@10': 19.9313, 'HR@20': 28.9722,
            'NDCG@5': 8.2287, 'NDCG@10': 10.5417, 'NDCG@20': 12.8210,
        },
        'STOSA': {
            'HR@5': 7.0495, 'HR@10': 14.3941, 'HR@20': 24.9871,
            'NDCG@5': 3.7174, 'NDCG@10': 6.0771, 'NDCG@20': 8.7241,
        },
        'DiffuRec_paper': {
            'HR@5': 17.9659, 'HR@10': 26.2647, 'HR@20': 36.7870,
            'NDCG@5': 12.1150, 'NDCG@10': 14.7909, 'NDCG@20': 17.4386,
        },
    },

    'steam': {
        # Steam baselines kept for completeness; not used unless you run Steam.
        'GRU4Rec': {
            'HR@5': 3.0124, 'HR@10': 5.4257, 'HR@20': 9.2319,
            'NDCG@5': 1.8293, 'NDCG@10': 2.6033, 'NDCG@20': 3.5572,
        },
        'Caser': {
            'HR@5': 3.6053, 'HR@10': 6.4940, 'HR@20': 10.9633,
            'NDCG@5': 2.1586, 'NDCG@10': 3.0846, 'NDCG@20': 4.2073,
        },
        'SASRec': {
            'HR@5': 4.7428, 'HR@10': 8.3763, 'HR@20': 13.6060,
            'NDCG@5': 2.8842, 'NDCG@10': 4.0489, 'NDCG@20': 5.3630,
        },
        'BERT4Rec': {
            'HR@5': 4.7391, 'HR@10': 7.9448, 'HR@20': 12.7332,
            'NDCG@5': 2.9708, 'NDCG@10': 4.0002, 'NDCG@20': 5.2027,
        },
        'ComiRec': {
            'HR@5': 2.2872, 'HR@10': 5.4358, 'HR@20': 10.3663,
            'NDCG@5': 1.0965, 'NDCG@10': 2.1053, 'NDCG@20': 3.3434,
        },
        'TiMiRec': {
            'HR@5': 6.0155, 'HR@10': 9.6697, 'HR@20': 14.8884,
            'NDCG@5': 3.8721, 'NDCG@10': 5.0446, 'NDCG@20': 6.3569,
        },
        'SVAE': {
            'HR@5': 3.2384, 'HR@10': 5.8275, 'HR@20': 7.9753,
            'NDCG@5': 1.8836, 'NDCG@10': 2.6881, 'NDCG@20': 3.2323,
        },
        'ACVAE': {
            'HR@5': 5.5825, 'HR@10': 9.2783, 'HR@20': 14.4846,
            'NDCG@5': 3.5429, 'NDCG@10': 4.7290, 'NDCG@20': 6.0374,
        },
        'STOSA': {
            'HR@5': 4.8546, 'HR@10': 8.5870, 'HR@20': 14.1107,
            'NDCG@5': 2.9220, 'NDCG@10': 4.1191, 'NDCG@20': 5.5072,
        },
        'DiffuRec_paper': {
            'HR@5': 6.6742, 'HR@10': 10.7520, 'HR@20': 16.6507,
            'NDCG@5': 4.2902, 'NDCG@10': 5.5981, 'NDCG@20': 7.0810,
        },
    },
}


# Display order — controls the sequence of rows in the main results table.
BASELINE_ORDER = [
    'GRU4Rec',
    'Caser',
    'SASRec',
    'BERT4Rec',
    'ComiRec',
    'TiMiRec',
    'SVAE',
    'ACVAE',
    'STOSA',
    'DiffuRec_paper',
]


# Pretty display names (with citation placeholder).
DISPLAY_NAMES = {
    'GRU4Rec':         'GRU4Rec~\\citep{hidasi2016session}',
    'Caser':           'Caser~\\citep{tang2018personalized}',
    'SASRec':          'SASRec~\\citep{kang2018self}',
    'BERT4Rec':        'BERT4Rec~\\citep{sun2019bert4rec}',
    'ComiRec':         'ComiRec~\\citep{cen2020controllable}',
    'TiMiRec':         'TiMiRec~\\citep{wang2022target}',
    'SVAE':            'SVAE~\\citep{sachdeva2019sequential}',
    'ACVAE':           'ACVAE~\\citep{xie2021adversarial}',
    'STOSA':           'STOSA~\\citep{fan2022sequential}',
    'DiffuRec_paper':  'DiffuRec~\\citep{li2023diffurec}',
}


def get_baselines(dataset_name):
    """Return the dict of baseline metrics for a given dataset, or None."""
    return LITERATURE_BASELINES.get(dataset_name)