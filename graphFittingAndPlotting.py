# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks
import ast
import re
import glob # .txt 파일을 찾기 위해 추가
import os   # 파일 이름을 다루기 위해 추가
import matplotlib.gridspec as gridspec # 복잡한 레이아웃을 위해 추가

# -----------------------------------------------------------
# 2. 통합 임피던스 계산 모델 (루프 외부로 이동)
# -----------------------------------------------------------
def calculate_Z_total(f, p, circuit_model_name, fit_p=None):
    w = 2 * np.pi * f
    j = 1j
    
    model_key = re.sub(r'#|\.txt|_data|\(.*\)', '', circuit_model_name).strip()
    is_dual_unit = ('RL-C + RL-C' in model_key)

    if fit_p is not None:
        if is_dual_unit:
            R1, R2, R_sense, L1_nominal, C1_nominal, a_L1, b_L1, a_C1, b_C1, c_C1, \
            L2_nominal, C2_nominal, a_L2, b_L2, a_C2, b_C2, c_C2 = fit_p
        else:
            R1, R_sense, L1_nominal, C1_nominal, a_L1, b_L1, a_C1, b_C1, c_C1 = fit_p
            R2, L2_nominal, C2_nominal = 0, 0, 0
            a_L2, b_L2, a_C2, b_C2, c_C2 = 0, 0, 0, 0, 0
    else: # 이상적인 경우 (파일의 공칭 값 사용)
        L1_nominal, L2_nominal = p['L1'], p.get('L2', 0)
        C1_nominal, C2_nominal = p['C1'], p.get('C2', 0)
        R1, R2      = p['R1'], p.get('R2', 0)
        R_sense     = p['R_sense']
        a_L1, b_L1, a_C1, b_C1, c_C1 = 0, 0, 0, 0, 0
        a_L2, b_L2, a_C2, b_C2, c_C2 = 0, 0, 0, 0, 0

    if L1_nominal > 1e-12:
        R_L1_freq_dependent = a_L1 * np.sqrt(f) + b_L1
        Z_L1 = R_L1_freq_dependent + j * w * L1_nominal
    else:
        Z_L1 = np.zeros_like(w, dtype=np.complex128)
    
    if C1_nominal > 1e-12:
        Y_C1_denominator = (j * w * C1_nominal) + (b_C1 * w) + c_C1
        Y_C1_denominator[Y_C1_denominator == 0] = 1e-18 
        Z_C1 = a_C1 + 1/Y_C1_denominator
    else:
        Z_C1 = np.full_like(w, np.inf, dtype=np.complex128)
    
    if 'RL-C + RL-C' in model_key:
        if L2_nominal > 1e-12:
            R_L2_freq_dependent = a_L2 * np.sqrt(f) + b_L2
            Z_L2 = R_L2_freq_dependent + j * w * L2_nominal
        else:
            Z_L2 = np.zeros_like(w, dtype=np.complex128)

        if C2_nominal > 1e-12:
            Y_C2_denominator = (j * w * C2_nominal) + (b_C2 * w) + c_C2
            Y_C2_denominator[Y_C2_denominator == 0] = 1e-18
            Z_C2 = a_C2 + 1/Y_C2_denominator
        else:
            Z_C2 = np.full_like(w, np.inf, dtype=np.complex128)
        
        Z_RL1 = R1 + Z_L1
        Z_RL2 = R2 + Z_L2
        Z_unit1_denom = Z_RL1 + Z_C1
        Z_unit1_denom[Z_unit1_denom == 0] = 1e-18
        Z_unit1 = (Z_RL1 * Z_C1) / Z_unit1_denom
        Z_unit2_denom = Z_RL2 + Z_C2
        Z_unit2_denom[Z_unit2_denom == 0] = 1e-18
        Z_unit2 = (Z_RL2 * Z_C2) / Z_unit2_denom
        Z_circuit = Z_unit1 + Z_unit2
    elif 'RL-C' in model_key: 
        Z_denom = R1 + Z_L1 + Z_C1
        Z_denom[Z_denom == 0] = 1e-18
        Z_circuit = ((R1 + Z_L1) * Z_C1) / Z_denom
    elif 'RC-L' in model_key: 
        Z_denom = R1 + Z_C1 + Z_L1
        Z_denom[Z_denom == 0] = 1e-18
        Z_circuit = ((R1 + Z_C1) * Z_L1) / Z_denom
    elif 'LC-R' in model_key: 
        Z_denom = Z_L1 + Z_C1 + R1
        Z_denom[Z_denom == 0] = 1e-18
        Z_circuit = ((Z_L1 + Z_C1) * R1) / Z_denom
    elif 'RLC' in model_key: Z_circuit = R1 + Z_L1 + Z_C1
    elif 'R-L-C' in model_key:
        Y_parallel = (1/R1 if R1 > 1e-12 else 0) + (1/Z_L1 if np.all(np.abs(Z_L1) > 1e-12) else 0) + (1/Z_C1 if np.all(np.abs(Z_C1) > 1e-12) else 0)
        Z_circuit = 1/Y_parallel if np.all(np.abs(Y_parallel) > 1e-12) else np.inf
    else:
        print(f"경고: '{model_key}' 이론식을 찾을 수 없습니다. 기본 RL-C 모델을 사용합니다.")
        Z_denom = R1 + Z_L1 + Z_C1
        Z_denom[Z_denom == 0] = 1e-18
        Z_circuit = ((R1 + Z_L1) * Z_C1) / Z_denom
            
    return Z_circuit + R_sense

# -----------------------------------------------------------
# 3. 피팅 모델 정의 (루프 외부로 이동)
# -----------------------------------------------------------
def objective_function_phase(p_fit, f_meas, phase_meas, p_nominal, circuit_model_name):
    Z_total_model = calculate_Z_total(f_meas, p_nominal, circuit_model_name, fit_p=p_fit)
    phase_model = np.angle(Z_total_model)
    return np.sum((np.sin(phase_meas - phase_model))**2)

def objective_function_amplitude(p_fit, f_meas, i_meas, p_nominal, circuit_model_name, p_phase):
    a_offset, vpk = p_fit
    Z_total_model = calculate_Z_total(f_meas, p_nominal, circuit_model_name, fit_p=p_phase)
    i_model = (1 / np.abs(Z_total_model) + a_offset) * vpk
    return np.sum((i_meas - i_model)**2)

# 오차율 계산 함수
def calculate_nrmse(y_true, y_pred):
    if len(y_true) == 0: return 0
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    range_y = np.max(y_true) - np.min(y_true)
    return rmse / range_y if range_y != 0 else 0

# 측정 데이터에서 공진점 찾는 함수
def find_peaks_from_measured(freqs, amps, window_size=5):
    peaks = []
    dips = []
    if len(amps) > 2 * window_size:
        for i in range(window_size, len(amps) - window_size):
            window = amps[i-window_size : i+window_size+1]
            if amps[i] >= np.max(window):
                peaks.append(freqs[i])
            if amps[i] <= np.min(window):
                dips.append(freqs[i])
    return sorted(list(set(peaks + dips)))

# --- 스크립트 시작 ---
all_files = glob.glob('*.txt')
output_summary_filename = "analysis_summary_all.txt"
all_files = [f for f in all_files if os.path.basename(f) != output_summary_filename]

order_map = {
    'RLC': '(a)',
    'LC-R': '(b)',
    'RC-L': '(c)',
    'RL-C': '(d)',
    'RL-C + RL-C 1개': '(e)',
    'RL-C + RL-C 3개': '(f)'
}

file_list = []
temp_files = {key: None for key in order_map}

for fname in all_files:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            model_key_from_file = re.sub(r'#|\.txt|_data|\(.*\)', '', first_line).strip()
            for key in order_map:
                if key == model_key_from_file:
                    temp_files[key] = fname
                    break
    except Exception:
        continue

for key in order_map:
    if temp_files[key]:
        file_list.append(temp_files[key])

if not file_list:
    print("분석할 .txt 파일을 찾을 수 없습니다. 스크립트와 동일한 폴더에 데이터 파일을 위치시켜주세요.")
    exit()

print(f"총 {len(file_list)}개의 데이터 파일을 찾았습니다. 지정된 순서대로 분석을 시작합니다.")

num_files = len(file_list)
nrows = 2
ncols = 3
fig = plt.figure(figsize=(8 * ncols, 6 * nrows))
outer_grid = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.3, hspace=0.1)


with open(output_summary_filename, 'w', encoding='utf-8') as summary_file:
    for i, input_filename in enumerate(file_list):
        print(f"\n{'='*70}")
        print(f"'{input_filename}' 파일 처리 중... ({i+1}/{len(file_list)})")
        print(f"{'='*70}")
        
        row = i // ncols
        col = i % ncols
        
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[i], hspace=0)
        ax1 = fig.add_subplot(inner_grid[0])
        ax2 = fig.add_subplot(inner_grid[1], sharex=ax1)
        
        try:
            # 1. 파일 읽기 및 데이터 파싱
            try:
                with open(input_filename, 'r', encoding='utf-8') as f: lines = f.readlines()
            except UnicodeDecodeError:
                with open(input_filename, 'r', encoding='cp949') as f: lines = f.readlines()
            circuit_model_name = lines[0].strip()
            params_line = lines[1].strip()
            params = ast.literal_eval(params_line)
            if 'R1' not in params: params['R1'] = params.get('R_circuit', params.get('R', 0))
            if 'C1' not in params: params['C1'] = params.get('C', 0)
            if 'L1' not in params: params['L1'] = params.get('L', 0)
            data = np.loadtxt(lines, delimiter=',', skiprows=2, usecols=(0, 1, 2, 3, 4))
            freq_measured, I_max_measured, I_err_measured, phase_measured, phase_err_measured = data.T
            print("데이터 읽기 완료.")
        
        except Exception as e:
            print(f"'{input_filename}' 파일 처리 중 오류 발생: {e}")
            summary_file.write(f"==== ERROR processing {os.path.basename(input_filename)} ====\n{e}\n\n{'-'*40}\n\n")
            ax1.text(0.5, 0.5, f"Error processing file:\n{os.path.basename(input_filename)}", ha='center', va='center', color='red')
            continue

        # 3. 피팅 실행
        model_key = re.sub(r'#|\.txt|_data|\(.*\)', '', circuit_model_name).strip()
        is_dual_unit = ('RL-C + RL-C' in model_key)
        if is_dual_unit:
            initial_guesses_phase = [params['R1'], params['R2'], params['R_sense'], params['L1'], params['C1'], 0.001, 0.1, 0.1, 0, 1e-6, params.get('L2',0), params.get('C2',0), 0.001, 0.1, 0.1, 0, 1e-6]
            bounds_phase = [(0, np.inf)] * 17
        else:
            initial_guesses_phase = [params['R1'], params['R_sense'], params['L1'], params['C1'], 0.001, 0.1, 0.1, 0, 1e-6]
            bounds_phase = [(0, np.inf)] * 9
        result_phase = minimize(objective_function_phase, initial_guesses_phase, args=(freq_measured, phase_measured, params, circuit_model_name), method='L-BFGS-B', bounds=bounds_phase)
        popt_phase = result_phase.x
        initial_guesses_amp = [0, params.get('Vpk', 1.0)]
        bounds_amp = [(-np.inf, np.inf), (0, np.inf)]
        result_amp = minimize(objective_function_amplitude, initial_guesses_amp, args=(freq_measured, I_max_measured, params, circuit_model_name, popt_phase), method='L-BFGS-B', bounds=bounds_amp)
        a_fitted, Vpk_fitted = result_amp.x

        # 4. 결과 계산
        f_theory = np.logspace(np.log10(freq_measured.min()), np.log10(freq_measured.max()), 2000)
        Z_total_ideal = calculate_Z_total(f_theory, params, circuit_model_name)
        I_max_ideal = params.get('Vpk', 1.0) / np.abs(Z_total_ideal)
        phase_ideal = np.angle(Z_total_ideal)
        Z_total_fitted_shape = calculate_Z_total(f_theory, params, circuit_model_name, fit_p=popt_phase)
        I_max_fitted = (1 / np.abs(Z_total_fitted_shape) + a_fitted) * Vpk_fitted
        phase_fitted = np.angle(Z_total_fitted_shape)
        
        prominence_threshold = (I_max_ideal.max() - I_max_ideal.min()) * 0.05
        peak_indices, _ = find_peaks(I_max_ideal, prominence=prominence_threshold)
        dip_indices, _ = find_peaks(-I_max_ideal, prominence=prominence_threshold)
        extrema_indices = np.sort(np.concatenate([peak_indices, dip_indices]))
        resonant_freqs_ideal = f_theory[extrema_indices]
        
        resonant_freqs_measured = find_peaks_from_measured(freq_measured, I_max_measured)

        I_ideal_interp = np.interp(freq_measured, f_theory, I_max_ideal)
        phase_ideal_interp = np.interp(freq_measured, f_theory, phase_ideal)
        I_fit_interp = np.interp(freq_measured, f_theory, I_max_fitted)
        phase_fit_interp = np.interp(freq_measured, f_theory, phase_fitted)
        err_I_ideal = calculate_nrmse(I_max_measured, I_ideal_interp)
        err_phase_ideal = calculate_nrmse(phase_measured, phase_ideal_interp)
        err_I_fit = calculate_nrmse(I_max_measured, I_fit_interp)
        err_phase_fit = calculate_nrmse(phase_measured, phase_fit_interp)

        # 5. 그래프 시각화
        label_text = f"({chr(ord('a') + i)})"
        ax1.text(-0.15, 1.05, label_text, transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top')
        
        ax1.errorbar(freq_measured, I_max_measured * 1000, yerr=I_err_measured * 1000, fmt='.', color='black',  markersize=2, label='Measured')
        ax1.plot(f_theory, I_max_ideal * 1000, '-', color='darkblue', label='Ideal')
        ax1.plot(f_theory, I_max_fitted * 1000, '--', color='darkred', label='Fit')
        
        if col == 0:
            ax1.set_ylabel('$I_{max}$ [mA]', fontsize=20)

        if model_key == 'RLC':
            ax1.legend(fontsize=20, frameon=False)
        
        ax1.grid(False, which="both")

        ax2.errorbar(freq_measured, np.rad2deg(phase_measured), yerr=np.rad2deg(phase_err_measured), fmt='.', color='black',  markersize=2)
        ax2.plot(f_theory, np.rad2deg(phase_ideal), '-', color='darkblue')
        ax2.plot(f_theory, np.rad2deg(phase_fitted), '--', color='darkred')
        
        if row == nrows - 1:
            ax2.set_xlabel('Frequency [Hz]', fontsize=20)
        if col == 0:
            ax2.set_ylabel('${\Delta\Phi}$ [°]', fontsize=20)
        ax2.grid(False, which="both")

        if len(resonant_freqs_ideal) > 0:
            for freq in resonant_freqs_ideal:
                ax1.axvline(x=freq, color='gray', linestyle='--', linewidth=1.2)
                ax2.axvline(x=freq, color='gray', linestyle='--', linewidth=1.2)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1.2)

        plt.setp(ax1.get_xticklabels(), visible=False)
        for ax in [ax1, ax2]:
            ax.set_xscale('log')
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.5, labelsize=16)
            ax.tick_params(axis='both', which='minor', direction='in', length=4, width=0.8)
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

        # 상세 결과 요약 .txt 파일에 추가
        summary_file.write(f"==== Analysis Summary for: {os.path.basename(input_filename)} ({label_text}) ====\n")
        summary_file.write(f"Circuit Model: {circuit_model_name}\n\n")

        summary_file.write("--- Nominal Parameters (from file) ---\n")
        for key, val in params.items():
            summary_file.write(f"  - {key:<10}: {val:.4e}\n")
        summary_file.write("\n")

        summary_file.write("--- Resonant Frequencies ---\n")
        # [수정] ValueError를 방지하기 위해 .size > 0 으로 확인
        summary_file.write(f"  - Ideal (calculated): {', '.join([f'{f:.1f} Hz' for f in resonant_freqs_ideal]) if resonant_freqs_ideal.size > 0 else 'None'}\n")
        summary_file.write(f"  - Measured (detected): {', '.join([f'{f:.1f} Hz' for f in resonant_freqs_measured]) if resonant_freqs_measured else 'None'}\n\n")

        summary_file.write("--- Fitted Parameters ---\n")
        fit_params_dict = {}
        if is_dual_unit:
            names = ["R1", "R2", "R_sense", "L1", "C1", "a_L1", "b_L1", "a_C1", "b_C1", "c_C1", "L2", "C2", "a_L2", "b_L2", "a_C2", "b_C2", "c_C2"]
        else:
            names = ["R1", "R_sense", "L1", "C1", "a_L1", "b_L1", "a_C1", "b_C1", "c_C1"]
        for k, v in zip(names, popt_phase): fit_params_dict[k] = v
        for name in names:
            val = fit_params_dict[name]
            if name in ["R1", "R2", "R_sense", "b_L1", "a_C1", "b_L2", "a_C2"]:
                summary_file.write(f"  - {name:<10}: {val:.4f} Ω\n")
            elif name in ["L1", "L2"]:
                summary_file.write(f"  - {name:<10}: {val:.4e} H\n")
            elif name in ["C1", "C2"]:
                summary_file.write(f"  - {name:<10}: {val:.4e} F\n")
            else:
                summary_file.write(f"  - {name:<10}: {val:.4e}\n")
        summary_file.write(f"  - {'Vpk_fit':<10}: {Vpk_fitted:.4f} V\n")
        summary_file.write(f"  - {'a_offset':<10}: {a_fitted:.4e}\n\n")
        summary_file.write("--- Error Analysis (NRMSE) ---\n")
        summary_file.write(f"  Ideal vs. Measured:\n")
        summary_file.write(f"    - Amplitude: {err_I_ideal:.2%}\n")
        summary_file.write(f"    - Phase:     {err_phase_ideal:.2%}\n")
        summary_file.write(f"  Fit vs. Measured:\n")
        summary_file.write(f"    - Amplitude: {err_I_fit:.2%}\n")
        summary_file.write(f"    - Phase:     {err_phase_fit:.2%}\n")
        summary_file.write(f"\n{'-'*40}\n\n")

# 비어있는 서브플롯을 숨깁니다.
for i in range(num_files, nrows * ncols):
    if i < len(outer_grid):
        fig.delaxes(fig.add_subplot(outer_grid[i]))

fig.tight_layout(rect=[0, 0.03, 1, 0.97])
output_image_filename = "analysis_grid.png"
plt.savefig(output_image_filename, dpi=1200, bbox_inches='tight')
print(f"\n{'='*70}")
print("모든 파일 분석이 완료되었습니다.")
print(f"분석 요약 파일: '{output_summary_filename}'")
print(f"통합 그래프 파일: '{output_image_filename}'")
plt.show()

