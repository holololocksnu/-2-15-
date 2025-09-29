import numpy as np
import matplotlib.pyplot as plt
from pydwf import DwfLibrary, DwfAnalogOutNode, DwfAnalogOutFunction, DwfAcquisitionMode, DwfState
import time
import re # 파일 이름 생성을 위해 re 모듈 추가

# 한글 폰트 설정
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rc('axes', unicode_minus=False)
except:
    pass

# -------------------------
# 1. 사용자 설정
# -------------------------
# 회로 정보
circuit_model_name = "# R-L-C"
circuit_params = {
    'R_sense': 10,    # 전류 감지 저항 (옴)
    'R_circuit': 10,  # 병렬 블록 내부 저항 (옴)
    'C': 20e-6,       # 커패시터 용량 (F)
    'L': 801e-6,        # 인덕터 용량 (H)
    'Vpk': 0.6,          # 파형의 피크 전압 (V)
    #'R1': 10,        # 직렬 저항 (옴)
    #'R2': 10,         # 병렬 저항 (옴)
    #'L1': 101e-6,       # 직렬 인덕터 (H)
    #'L2': 101e-6,       # 병렬 인덕터 (H)
    #'C1': 0.1e-6,      # 직렬 커패시터 (F)
    #'C2': 0.1e-6       # 병렬 커패시터 (F)
}

# 주파수 스윕 설정
start_freq = 0.1e3  # 시작 주파수 (0.1 kHz)
end_freq = 50e3     # 종료 주파수 (50 kHz)
num_steps = 50     # 측정할 주파수 포인트 개수

# 측정 설정
sample_rate = 1e6   # 샘플링 속도 (Hz)
num_cycles_to_capture = 40 # 넉넉하게 40주기 캡처
num_cycles_to_stabilize = 10 # 과도 응답을 고려하여 분석에서 제외할 초기 주기 수

# -----------------------------------------------------------
# 2. 실험 및 분석 준비
# -----------------------------------------------------------
# 최종 분석 결과를 저장할 리스트
recalculated_freqs = []
recalculated_I_max = []
recalculated_I_err = []
recalculated_phases = []
recalculated_phase_err = []

freq_steps = np.logspace(np.log10(start_freq), np.log10(end_freq), num=num_steps)
dwf = DwfLibrary()
print("병렬 RLC 회로 주파수 응답 측정 및 분석을 시작합니다.")
print(f"주파수 범위: {start_freq:.0f} Hz ~ {end_freq:.0f} Hz ({num_steps} 단계)")

# -----------------------------------------------------------
# 3. 주파수 스윕, 측정 및 실시간 분석 루프
# -----------------------------------------------------------
try:
    with dwf.deviceControl.open(-1) as device:
        wavegen, osc = device.analogOut, device.analogIn
        node = DwfAnalogOutNode.Carrier

        # Wavegen 기본 설정
        wavegen.nodeEnableSet(0, node, True)
        wavegen.nodeFunctionSet(0, node, DwfAnalogOutFunction.Sine)
        wavegen.nodeAmplitudeSet(0, node, circuit_params['Vpk'])
        wavegen.nodeOffsetSet(0, node, 0.0)
        
        for freq in freq_steps:
            # --- 3.1. 데이터 수집 ---
            print(f"\n===== 현재 주파수: {freq:.0f} Hz 측정 중... =====")
            wavegen.nodeFrequencySet(0, node, freq)
            wavegen.configure(0, True)
            time.sleep(0.3) # 안정화 시간
            
            osc.reset()
            osc.channelEnableSet(0, True)
            osc.channelEnableSet(1, True)
            osc.channelRangeSet(0, 5.0)
            osc.channelRangeSet(1, 5.0)
            osc.acquisitionModeSet(DwfAcquisitionMode.Record)
            osc.frequencySet(sample_rate)
            
            record_duration = num_cycles_to_capture / freq
            osc.recordLengthSet(record_duration)
            osc.configure(False, True)
            
            all_data_ch1, all_data_ch2 = [], []
            total_samples_to_acquire = int(record_duration * sample_rate)
            while len(np.concatenate(all_data_ch1 if all_data_ch1 else [[]])) < total_samples_to_acquire:
                device_state = osc.status(True)
                cAvailable, _, _ = osc.statusRecord()
                if cAvailable > 0:
                    all_data_ch1.append(np.array(osc.statusData(0, cAvailable)))
                    all_data_ch2.append(np.array(osc.statusData(1, cAvailable)))
                if device_state == DwfState.Done and cAvailable == 0:
                    break
            
            if not all_data_ch1 or not all_data_ch2:
                print(f"!!! 경고: {freq:.0f} Hz에서 데이터 수집 실패. 건너뜁니다.")
                continue
                
            v0_full = np.concatenate(all_data_ch1)
            v1_full = np.concatenate(all_data_ch2)
            
            # 과도 응답 구간 데이터 제거
            samples_to_discard = int(num_cycles_to_stabilize / freq * sample_rate)
            if len(v0_full) > samples_to_discard:
                v0 = v0_full[samples_to_discard:]
                v1 = v1_full[samples_to_discard:]
                print(f"  - 초기 {num_cycles_to_stabilize} 주기 데이터({samples_to_discard} 샘플)를 분석에서 제외합니다.")
            else:
                v0, v1 = v0_full, v1_full
                
            t = np.arange(len(v0)) / sample_rate

            # --- 3.2. 데이터 분석 (안정화된 데이터로 수행) ---
            V = v0
            I = v1 / circuit_params['R_sense']

            # [수정] FFT 기반 진폭 및 위상차 계산
            n_splits = 10
            split_size = len(t) // n_splits
            amplitudes_in_splits = []
            phases_in_splits_rad = []
            
            if split_size > 10: # FFT를 위해 최소한의 데이터 포인트 확보
                for i in range(n_splits):
                    start_idx = i * split_size
                    end_idx = start_idx + split_size
                    if i == n_splits - 1: end_idx = len(t)

                    V_split, I_split = V[start_idx:end_idx], I[start_idx:end_idx]
                    t_split = t[start_idx:end_idx]

                    if len(t_split) > 1:
                        # FFT 수행
                        fft_V = np.fft.fft(V_split)
                        fft_I = np.fft.fft(I_split)
                        fft_freq = np.fft.fftfreq(len(t_split), d=(t_split[1]-t_split[0]))
                        idx = np.argmin(np.abs(fft_freq - freq))
                        
                        # 진폭 계산
                        amp_I = (np.abs(fft_I[idx]) * 2) / len(I_split)
                        amplitudes_in_splits.append(amp_I)
                        
                        # 위상차 계산
                        phase_V = np.angle(fft_V[idx])
                        phase_I = np.angle(fft_I[idx])
                        phase_diff = np.arctan2(np.sin(phase_V - phase_I), np.cos(phase_V - phase_I))
                        phases_in_splits_rad.append(phase_diff)

            if amplitudes_in_splits: # 구간 분할 분석이 성공했을 경우
                I_max = np.mean(amplitudes_in_splits)
                I_err = np.std(amplitudes_in_splits)
                phase_diff_rad = np.mean(phases_in_splits_rad)
                phase_err_rad = np.std(phases_in_splits_rad)
            else: # 데이터가 너무 짧아 분할이 불가능한 경우, 전체 데이터로 한 번만 계산
                if len(t) > 1:
                    fft_V = np.fft.fft(V)
                    fft_I = np.fft.fft(I)
                    fft_freq = np.fft.fftfreq(len(t), d=(t[1]-t[0]))
                    idx = np.argmin(np.abs(fft_freq - freq))
                    
                    I_max = (np.abs(fft_I[idx]) * 2) / len(I)
                    phase_V = np.angle(fft_V[idx])
                    phase_I = np.angle(fft_I[idx])
                    phase_diff_rad = np.arctan2(np.sin(phase_V - phase_I), np.cos(phase_V - phase_I))
                else:
                    I_max = 0
                    phase_diff_rad = 0
                I_err = 0
                phase_err_rad = 0
                
            # --- 3.3. 분석 결과 저장 ---
            recalculated_freqs.append(freq)
            recalculated_I_max.append(I_max)
            recalculated_I_err.append(I_err)
            recalculated_phases.append(phase_diff_rad)
            recalculated_phase_err.append(phase_err_rad)
            print(f"분석 완료: I_max = {I_max*1000:.3f}±{I_err*1000:.3f} mA, Phase = {np.degrees(phase_diff_rad):.2f}±{np.degrees(phase_err_rad):.2f}°")

finally:
    # --- 3.4. 장비 리셋 ---
    try:
        with dwf.deviceControl.open(-1) as device:
            device.analogOut.reset(0)
            device.analogIn.reset()
            print("\n실험 완료. 장비가 리셋되었습니다.")
    except Exception as e:
        print(f"장비 리셋 중 오류 발생: {e}")

# -----------------------------------------------------------
# 4. 최종 결과 파일 저장
# -----------------------------------------------------------
if recalculated_freqs:
    # circuit_model_name에서 유효한 파일 이름을 생성
    try:
        # 특수 문자를 제거하고 공백을 밑줄로 바꿔서 안전한 파일 이름 생성
        output_filename = f'{circuit_model_name}_data.txt'

        print(f"\n분석된 최종 데이터를 '{output_filename}' 파일로 저장합니다...")
        with open(output_filename, 'w') as f_out:
            f_out.write(f"{circuit_model_name}\n")
            # 파라미터를 딕셔너리 형태로 저장
            f_out.write(f"{circuit_params}\n")
            
            for data_tuple in zip(
                recalculated_freqs, 
                recalculated_I_max, 
                recalculated_I_err, 
                recalculated_phases, 
                recalculated_phase_err
            ):
                # 각 값을 과학적 표기법(e-notation)으로 저장
                f_out.write(f"{data_tuple[0]:.6e},{data_tuple[1]:.6e},{data_tuple[2]:.6e},{data_tuple[3]:.6e},{data_tuple[4]:.6e}\n")
        print("데이터 저장이 완료되었습니다.")
        
        # -----------------------------------------------------------
        # 5. 간단한 결과 그래프 표시
        # -----------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('RLC 회로 주파수 응답 측정 결과', fontsize=16)

        # 전류 그래프
        ax1.errorbar(recalculated_freqs, np.array(recalculated_I_max) * 1000, 
                     yerr=np.array(recalculated_I_err) * 1000,
                     fmt='o', capsize=3, markersize=4, label='측정된 I_max')
        ax1.set_ylabel('전류 최댓값 (I_max) [mA]')
        ax1.set_xscale('log')
        ax1.grid(True, which="both", ls="--")
        ax1.legend()
        
        # 위상 그래프
        ax2.errorbar(recalculated_freqs, np.degrees(recalculated_phases), 
                     yerr=np.degrees(recalculated_phase_err),
                     fmt='o', capsize=3, markersize=4, color='green', label='측정된 위상차')
        ax2.set_xlabel('주파수 (Frequency) [Hz]')
        ax2.set_ylabel('위상차 (Phase) [°]')
        ax2.set_xscale('log')
        ax2.grid(True, which="both", ls="--")
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        print(f"'{output_filename}' 파일 저장 또는 그래프 생성 중 오류 발생: {e}")
else:
    print("\n측정 및 분석된 데이터가 없어 파일을 저장할 수 없습니다.")

