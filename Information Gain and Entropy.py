import pandas as pd
import numpy as np

data_set = {
    'qwlook': ['s', 's', 's', 'z', 'z', 'z', 'z', 's', 's', 'z', 's', 'z', 's', 'z'],
    'tamp': ['h', 'h', 'h', 'c', 'c', 'c', 'c', 'h', 'c', 'h', 'c', 'h', 'h', 'c'],
    'hid': ['t', 't', 'n', 't', 'n', 'n', 't', 'n', 't', 't', 't', 'n', 'n', 'n'],
    'winx': ['g', 'o', 'o', 'o', 'g', 'o', 'o', 'o', 'g', 'g', 'g', 'o', 'o', 'g'],
    'yes_no': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

data_x = pd.DataFrame.from_dict(data_set)
print(data_x)

counst_lg = data_x['winx'].value_counts()['g']
col1 = len(data_x[(data_x['winx'] == 'g') & (data_x['yes_no'] == 'yes')])
print(col1)


def entropy(valuse_x, valuse_y, column_name, data_f, column_name2, valuse_1x, valuse_2y, var1, var2):
    l = len(data_f[column_name])
    counst_y = data_f[column_name].value_counts()[valuse_x]
    counst_x = data_f[column_name].value_counts()[valuse_y]
    len_y = data_f[column_name2].value_counts()[valuse_1x]
    len_g = data_f[column_name2].value_counts()[valuse_2y]

    p_y = counst_y / l
    p_x = counst_x / l
    enp = 0
    if p_y > 0:
        enp -= p_y * np.log2(p_y)
    if p_x > 0:
        enp -= p_x * np.log2(p_x)

    # الدالة الفرعية لحساب entropy لكل قيمة
    def ent(colum_name2, valuse, var1, var2):
        counst_lg = data_f[colum_name2].value_counts()[valuse]
        #داله معرفه عدد valuse colum1 مقابل valuse column2في كل حاله
        col1 = len(data_f[(data_f[colum_name2] == valuse) & (data_f[column_name] == var1)])
        col2 = len(data_f[(data_f[colum_name2] == valuse) & (data_f[column_name] == var2)])

        p_y = col1 / counst_lg if counst_lg != 0 else 0
        p_x = col2 / counst_lg if counst_lg != 0 else 0
        # حساب Entropy لكل valuse
        enp_v = 0
        if p_y > 0:
            enp_v -= p_y * np.log2(p_y)
        if p_x > 0:
            enp_v -= p_x * np.log2(p_x)
        return enp_v
    #حساب Entropy لكل valuse
    varx_1 = ent(column_name2, valuse_1x, var1, var2)
    vary_2 = ent(column_name2, valuse_2y, var1, var2)

    #  حساب Information Gai
    information_gan = enp - ((len_y / l) * varx_1) - ((len_g / l) * vary_2)

    return information_gan


qwlook = entropy('no', 'yes', 'yes_no', data_x, 'qwlook', 's', 'z', 'yes', 'no')
tamp = entropy('no', 'yes', 'yes_no', data_x, 'tamp', 'h', 'c', 'yes', 'no')
hid = entropy('no', 'yes', 'yes_no', data_x, 'hid', 't', 'n', 'yes', 'no')
winx = entropy('no', 'yes', 'yes_no', data_x, 'winx', 'g', 'o', 'yes', 'no')
print( qwlook,tamp,hid,winx)
