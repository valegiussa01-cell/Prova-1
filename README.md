import pandas as pd
import numpy as np
import pvlib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class PVSystem:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude
        self.tilt = 30
        self.azimuth_pvlib = 180
        self.num_panels = 5  
        self.panel_power_stc = 435  
        self.panel_efficiency_stc = 0.218  
        self.temperature_coefficient_pmax = -0.0034  
        
        self.NOCT_datasheet = 43
        self.roof_delta = 2
        self.NOCT_effective = self.NOCT_datasheet + self.roof_delta
        
        self.module_area = 1.762 * 1.134
        self.total_area = self.num_panels * self.module_area
        self.PR_ref = 0.94
        self.inverter_eff = 0.96
        self.installed_power_kw = (self.num_panels * self.panel_power_stc) / 1000
        
        if self.latitude >= 50:
            self.clima_type = "NORDICO"
            self.pr_a, self.pr_b = 0.88, 0.12
        elif self.latitude >= 40:
            self.clima_type = "MEDITERRANEO"
            self.pr_a, self.pr_b = 0.90, 0.10
        else:
            self.clima_type = "EQUATORIALE"
            self.pr_a, self.pr_b = 0.92, 0.08
       
        print(f"\n{'='*60}")
        print(f"SISTEMA: {self.installed_power_kw:.2f} kWp | PR_ref: {self.PR_ref}")
        print(f"Coordinate: {self.latitude}N, {self.longitude}E")
        print(f"Clima: {self.clima_type}")
        print(f"Formula PR: {self.pr_a} + {self.pr_b} * kt")
        print(f"Installazione: Roof Added (delta = +{self.roof_delta}C)")
        print(f"NOCT effettivo: {self.NOCT_datasheet}C (datasheet) + {self.roof_delta}C = {self.NOCT_effective}C")
        print(f"{'='*60}")

def build_tmy_from_multiyear(system):
    print(f"Download dati PVGIS (2005-2020) per {system.latitude}N, {system.longitude}E...")
    try:
        res = pvlib.iotools.get_pvgis_hourly(
            latitude=system.latitude, longitude=system.longitude,
            start=2005, end=2020, surface_tilt=system.tilt,
            surface_azimuth=system.azimuth_pvlib, components=True,
            usehorizon=True, raddatabase='PVGIS-SARAH3'
        )
        data = res[0].copy()
       
        if 'time' in data.columns:
            data = data.drop(columns=['time'])
           
        data = data[~((data.index.month == 2) & (data.index.day == 29))]
       
        data['poa_global'] = data['poa_direct'] + data['poa_sky_diffuse'] + data['poa_ground_diffuse']
       
        p_col = [c for c in data.columns if c in ['P', 'P_pv']]
        if p_col:
            data['pvgis_prod_kwh'] = data[p_col[0]] / 1000
        else:
            data['pvgis_prod_kwh'] = (data['poa_global'] / 1000) * system.installed_power_kw * 0.84

        tmy_data = data.groupby([data.index.month, data.index.day, data.index.hour]).mean()
       
        datetime_index = pd.date_range(start='2015-01-01', periods=8760, freq='h')
        result = pd.DataFrame(index=datetime_index)
       
        result['poa_global'] = tmy_data['poa_global'].values
        result['temp_air'] = tmy_data['temp_air'].values
        result['pvgis_ref_kwh'] = tmy_data['pvgis_prod_kwh'].values
       
        return result
    except Exception as e:
        print(f"Errore: {e}")
        return None

def calculate_daily_data(system, hourly_data):
    if hourly_data is None: 
        return None
   
    hourly_data['g_avg'] = hourly_data['poa_global']
    hourly_data['kt'] = np.clip(hourly_data['g_avg'] / 1000, 0, 1)
    
    hourly_data['pr_dyn'] = system.PR_ref * (system.pr_a + system.pr_b * hourly_data['kt'])
   
    hourly_data['t_cell'] = hourly_data['temp_air'] + (system.NOCT_effective - 20) * (hourly_data['g_avg'] / 800)
    hourly_data['eff'] = system.panel_efficiency_stc * (1 + system.temperature_coefficient_pmax * (hourly_data['t_cell'] - 25))
    
    hourly_data['efficiency_system'] = hourly_data['eff'] * hourly_data['pr_dyn'] * system.inverter_eff * 100
   
    hourly_data['prod_h'] = (hourly_data['poa_global'] / 1000) * system.total_area * hourly_data['eff'] * hourly_data['pr_dyn'] * system.inverter_eff
   
    daily = hourly_data.resample('D').agg({
        'prod_h': 'sum',
        'pvgis_ref_kwh': 'sum',
        'poa_global': 'sum',
        'temp_air': 'mean',
        't_cell': 'mean',
        'pr_dyn': 'mean',
        'eff': 'mean',
        'efficiency_system': 'mean'
    }).rename(columns={
        'prod_h': 'prod_mio', 
        'pvgis_ref_kwh': 'prod_pvgis',
        'poa_global': 'rad_whm2',
        'temp_air': 'temp_aria',
        't_cell': 'temp_cella',
        'pr_dyn': 'pr_medio',
        'eff': 'efficienza_cella',
        'efficiency_system': 'efficienza_sistema_perc'
    })
    
    daily['rad_kwhm2'] = daily['rad_whm2'] / 1000
    daily['error_abs'] = daily['prod_mio'] - daily['prod_pvgis']
    daily['error_rel'] = (daily['error_abs'] / daily['prod_pvgis']) * 100
    daily['error_squared'] = (daily['prod_mio'] - daily['prod_pvgis']) ** 2
    daily['abs_error_rel'] = np.abs(daily['error_rel'])
   
    return daily

def calculate_metrics(daily_df, monthly_df):
    actual_daily = daily_df['prod_pvgis'].values
    predicted_daily = daily_df['prod_mio'].values
    
    mape_daily = np.mean(np.abs((actual_daily - predicted_daily) / actual_daily)) * 100
    mbe_daily = np.mean(predicted_daily - actual_daily)
    mbe_rel_daily = (mbe_daily / np.mean(actual_daily)) * 100
    rmse_daily = np.sqrt(np.mean((predicted_daily - actual_daily) ** 2))
    rmse_rel_daily = (rmse_daily / np.mean(actual_daily)) * 100
    nrmse_daily = rmse_daily / (np.max(actual_daily) - np.min(actual_daily)) * 100
    
    actual_monthly = monthly_df['prod_pvgis'].values
    predicted_monthly = monthly_df['prod_mio'].values
    
    mape_monthly = np.mean(np.abs((actual_monthly - predicted_monthly) / actual_monthly)) * 100
    mbe_monthly = np.mean(predicted_monthly - actual_monthly)
    mbe_rel_monthly = (mbe_monthly / np.mean(actual_monthly)) * 100
    rmse_monthly = np.sqrt(np.mean((predicted_monthly - actual_monthly) ** 2))
    rmse_rel_monthly = (rmse_monthly / np.mean(actual_monthly)) * 100
    nrmse_monthly = rmse_monthly / (np.max(actual_monthly) - np.min(actual_monthly)) * 100
    
    ss_res = np.sum((actual_monthly - predicted_monthly) ** 2)
    ss_tot = np.sum((actual_monthly - np.mean(actual_monthly)) ** 2)
    r2_monthly = 1 - (ss_res / ss_tot)
    
    return {
        'daily': {
            'MAPE (%)': mape_daily,
            'MBE (kWh)': mbe_daily,
            'MBE_rel (%)': mbe_rel_daily,
            'RMSE (kWh)': rmse_daily,
            'RMSE_rel (%)': rmse_rel_daily,
            'nRMSE (%)': nrmse_daily
        },
        'monthly': {
            'MAPE (%)': mape_monthly,
            'MBE (kWh)': mbe_monthly,
            'MBE_rel (%)': mbe_rel_monthly,
            'RMSE (kWh)': rmse_monthly,
            'RMSE_rel (%)': rmse_rel_monthly,
            'nRMSE (%)': nrmse_monthly,
            'R²': r2_monthly
        }
    }

def print_metrics_analysis(metrics):
    print(f"\n{'='*90}")
    print(f"METRICHE DI ERRORE - VALIDAZIONE MODELLO")
    print(f"{'='*90}")
    
    print(f"\nMETRICHE GIORNALIERE:")
    print(f"{'-'*60}")
    print(f"MAPE:  {metrics['daily']['MAPE (%)']:.3f}%")
    print(f"MBE:   {metrics['daily']['MBE (kWh)']:.4f} kWh  ({metrics['daily']['MBE_rel (%)']:+.3f}%)")
    print(f"RMSE:  {metrics['daily']['RMSE (kWh)']:.4f} kWh  ({metrics['daily']['RMSE_rel (%)']:.3f}%)")
    print(f"nRMSE: {metrics['daily']['nRMSE (%)']:.3f}%")
    
    print(f"\nCOMMENTO METRICHE GIORNALIERE:")
    if metrics['daily']['MAPE (%)'] < 5:
        print(f"   MAPE = {metrics['daily']['MAPE (%)']:.3f}% -> ECCELLENTE")
    elif metrics['daily']['MAPE (%)'] < 10:
        print(f"   MAPE = {metrics['daily']['MAPE (%)']:.3f}% -> ACCETTABILE")
    else:
        print(f"   MAPE = {metrics['daily']['MAPE (%)']:.3f}% -> MIGLIORABILE")
    
    print(f"\nMETRICHE MENSILI:")
    print(f"{'-'*60}")
    print(f"MAPE:   {metrics['monthly']['MAPE (%)']:.3f}%")
    print(f"MBE:    {metrics['monthly']['MBE (kWh)']:.4f} kWh  ({metrics['monthly']['MBE_rel (%)']:+.3f}%)")
    print(f"RMSE:   {metrics['monthly']['RMSE (kWh)']:.4f} kWh  ({metrics['monthly']['RMSE_rel (%)']:.3f}%)")
    print(f"nRMSE:  {metrics['monthly']['nRMSE (%)']:.3f}%")
    print(f"R²:     {metrics['monthly']['R²']:.5f}")
    
    print(f"\nVALUTAZIONE COMPLESSIVA:")
    score = 0
    if metrics['monthly']['MAPE (%)'] < 1: score += 3
    if abs(metrics['monthly']['MBE_rel (%)']) < 0.2: score += 3
    if metrics['monthly']['R²'] > 0.99: score += 4
    
    if score >= 9:
        print(f"   PUNTEGGIO: {score}/10 -> MODELLO ECCELLENTE")
    elif score >= 7:
        print(f"   PUNTEGGIO: {score}/10 -> MODELLO BUONO")
    else:
        print(f"   PUNTEGGIO: {score}/10 -> MODELLO MIGLIORABILE")

def print_report(daily_df):
    monthly = daily_df.resample('M').agg({
        'prod_mio': 'sum',
        'prod_pvgis': 'sum',
        'rad_kwhm2': 'sum',
        'temp_aria': 'mean',
        'temp_cella': 'mean',
        'efficienza_cella': 'mean',
        'efficienza_sistema_perc': 'mean',
        'pr_medio': 'mean',
        'error_rel': 'mean'
    })
    
    print(f"\n{'='*90}")
    print(f"REPORT MENSILE - PRODUZIONE")
    print(f"{'='*90}")
    print(f"\n{'MESE':<10} | {'MIO MODELLO':<12} | {'PVGIS':<12} | {'DIFF %':<8}")
    print("-" * 55)
    
    for i, row in enumerate(monthly.itertuples(), 1):
        diff = ((row.prod_mio - row.prod_pvgis) / row.prod_pvgis) * 100
        print(f"{i:<10} | {row.prod_mio:>11.1f} | {row.prod_pvgis:>11.1f} | {diff:>7.2f}%")
   
    total_mio = daily_df['prod_mio'].sum()
    total_pvgis = daily_df['prod_pvgis'].sum()
    total_diff = ((total_mio - total_pvgis) / total_pvgis) * 100
    print("-" * 55)
    print(f"TOTALE     | {total_mio:>11.1f} | {total_pvgis:>11.1f} | {total_diff:>7.2f}%")
    
    print(f"\n{'='*90}")
    print(f"REPORT MENSILE - EFFICIENZE E TEMPERATURE")
    print(f"{'='*90}")
    print(f"\n{'MESE':<10} | {'EFF. CELLA':<12} | {'EFF. SISTEMA':<12} | {'PR':<8} | {'T_CELLA':<10} | {'T_ARIA':<10}")
    print("-" * 75)
    
    for i, row in enumerate(monthly.itertuples(), 1):
        eff_cella_pct = row.efficienza_cella * 100
        print(f"{i:<10} | {eff_cella_pct:>11.1f}% | {row.efficienza_sistema_perc:>11.1f}% | {row.pr_medio:>7.3f} | {row.temp_cella:>9.1f}C | {row.temp_aria:>9.1f}C")
    
    print(f"\n{'='*90}")
    print(f"STATISTICHE ANNUALI")
    print(f"{'='*90}")
    installed_kw = 2.175
    print(f"Radiazione annuale: {monthly['rad_kwhm2'].sum():.1f} kWh/m²")
    print(f"Produzione specifica: {total_mio / installed_kw:.1f} kWh/kWp")
    print(f"Performance Ratio medio: {monthly['pr_medio'].mean():.3f}")
    print(f"Temperatura cella media: {monthly['temp_cella'].mean():.1f}C")
    print(f"Efficienza sistema media: {monthly['efficienza_sistema_perc'].mean():.1f}%")
    print(f"Fattore di capacita: {(total_mio / (installed_kw * 8760)) * 100:.1f}%")
    
    metrics = calculate_metrics(daily_df, monthly)
    print_metrics_analysis(metrics)
    
    return metrics

def export_to_excel(daily_df, metrics, filename):
    print(f"\nEsportazione dati in Excel...")
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            daily_export = daily_df.copy()
            daily_export.index = daily_export.index.strftime('%Y-%m-%d')
            daily_export.to_excel(writer, sheet_name='Dati_Giornalieri')
            
            monthly = daily_df.resample('M').agg({
                'prod_mio': 'sum',
                'prod_pvgis': 'sum',
                'rad_kwhm2': 'sum',
                'temp_aria': 'mean',
                'temp_cella': 'mean',
                'efficienza_sistema_perc': 'mean',
                'pr_medio': 'mean',
                'error_rel': 'mean'
            })
            monthly.index = monthly.index.strftime('%Y-%m')
            monthly.to_excel(writer, sheet_name='Riepilogo_Mensile')
            
            metrics_df = pd.DataFrame({
                'Metrica': ['MAPE (%)', 'MBE (kWh)', 'MBE_rel (%)', 'RMSE (kWh)', 'RMSE_rel (%)', 'nRMSE (%)', 'R²'],
                'Giornaliero': [
                    metrics['daily']['MAPE (%)'],
                    metrics['daily']['MBE (kWh)'],
                    metrics['daily']['MBE_rel (%)'],
                    metrics['daily']['RMSE (kWh)'],
                    metrics['daily']['RMSE_rel (%)'],
                    metrics['daily']['nRMSE (%)'],
                    '-'
                ],
                'Mensile': [
                    metrics['monthly']['MAPE (%)'],
                    metrics['monthly']['MBE (kWh)'],
                    metrics['monthly']['MBE_rel (%)'],
                    metrics['monthly']['RMSE (kWh)'],
                    metrics['monthly']['RMSE_rel (%)'],
                    metrics['monthly']['nRMSE (%)'],
                    metrics['monthly']['R²']
                ]
            })
            metrics_df.to_excel(writer, sheet_name='Metriche_Errore', index=False)
            
        print(f"File Excel salvato: {filename}")
    except Exception as e:
        print(f"Errore nel salvataggio Excel: {e}")

def print_daily_summary(daily_df, num_days=10):
    print(f"\n{'='*100}")
    print(f"RIEPILOGO PRIMI {num_days} GIORNI")
    print(f"{'='*100}")
    
    print(f"\n{'DATA':<12} | {'PROD MIO':<10} | {'PROD PVGIS':<10} | {'ERRORE %':<9} | {'RAD(kWh/m²)':<12} | {'EFF SIST%':<10}")
    print("-" * 80)
    
    for i in range(min(num_days, len(daily_df))):
        row = daily_df.iloc[i]
        date = daily_df.index[i].strftime('%Y-%m-%d')
        errore_pct = ((row['prod_mio'] - row['prod_pvgis']) / row['prod_pvgis']) * 100
        print(f"{date:<12} | {row['prod_mio']:>9.2f} | {row['prod_pvgis']:>9.2f} | {errore_pct:>8.2f} | {row['rad_kwhm2']:>10.2f} | {row['efficienza_sistema_perc']:>9.1f}%")

def main():
    latitude = 53.383
    longitude = -6.592
    
    print(f"\n{'#'*60}")
    print(f"ANALISI PER LOCALITA: LAT {latitude}N, LON {longitude}E")
    print(f"{'#'*60}")
    
    sys = PVSystem(latitude=latitude, longitude=longitude)
    data = build_tmy_from_multiyear(sys)
    
    if data is not None:
        daily = calculate_daily_data(sys, data)
        metrics = print_report(daily)
        print_daily_summary(daily, num_days=15)
        
        filename = f'produzione_fotovoltaico_{latitude}_{longitude}.xlsx'
        export_to_excel(daily, metrics, filename)
        
        print(f"\n{'='*60}")
        print(f"ANALISI COMPLETATA")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()
