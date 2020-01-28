# Import modules
import numpy as np
import matplotlib.pyplot as plt
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import BasicAer
from qiskit.extensions import IdGate
from IPython.display import display, Math, Latex, clear_output

# Function for showing a state vector in a nice way, from QuantumInspire website
def format_vector(state_vector, decimal_precision=7):
    """ Format the state vector into a LaTeX formatted string.

    Args:
        state_vector (list or array): The state vector with complex
                                      values e.g. [-1, 2j+1].

    Returns:
        str: The LaTeX format.
    """
    result = []
    epsilon = 1/pow(10, decimal_precision)
    bit_length = (len(state_vector) - 1).bit_length()
    for index, complex_value in enumerate(state_vector):
        has_imag_part = np.round(complex_value.imag, decimal_precision) != 0.0
        value = complex_value if has_imag_part else complex_value.real
        value_round = np.round(value, decimal_precision)
        if np.abs(value_round) < epsilon:
            continue

        binary_state = '{0:0{1}b}'.format(index, bit_length)
        result.append(r'{0:+2g}\left\lvert {1}\right\rangle '.format(value_round, binary_state))
    return ''.join(result)

# Function for running all possible inputs through a circuit, from QuantumInspire website
def run_circuit(q_circuit, q_register, number_of_qubits=None, backend_name='statevector_simulator'):
    """ Run a circuit on all base state vectors and show the output.

    Args:
        q_circuit (QuantumCircuit):
        q_register (QuantumRegister)
        number_of_qubits (int or None): The number of qubits.
        backend (str): ...
    """
    if not isinstance(number_of_qubits, int):
        number_of_qubits = q_register.size

    if q_register.size != number_of_qubits:
        warnings.warn('incorrect register size?')

    latex_text = r'\mathrm{running\ circuit\ on\ set\ of\ basis\ states:}'
    display(Math(latex_text))

    base_states = 2 ** number_of_qubits
    backend = BasicAer.get_backend(backend_name)
    for base_state in range(base_states):
        pre_circuit = QuantumCircuit(q_register)
        state = base_state
        for kk in range(number_of_qubits):
            if state % 2 == 1:
                pre_circuit.x(q_register[kk])
            state = state // 2    

        input_state = r'\left\lvert{0:0{1}b}\right\rangle'.format(base_state, number_of_qubits)
        circuit_total = pre_circuit.combine(q_circuit)
        job = execute(circuit_total, backend=backend)
        output_state = job.result().get_statevector(circuit_total)

        latex_text = input_state + r'\mathrm{transforms\ to}: ' + format_vector(output_state,decimal_precision=2)
        display(Math(latex_text))
        
# Adjusted run_circuit function to only try with zeros as input
def run_circuit0(q_circuit, number_of_qubits=None, backend_name='statevector_simulator'):
    """ Run a circuit on base state zerovector and show the output.

    Args:
        q_circuit (QuantumCircuit):
        number_of_qubits (int or None): The number of qubits.
        backend (str): ...
    """
    latex_text = r'\mathrm{running\ circuit\ on\ zerostate:}'
    display(Math(latex_text))

    backend = BasicAer.get_backend(backend_name)
    
    job = execute(q_circuit, backend=backend)
    output_state = job.result().get_statevector(q_circuit)
    input_state = r'\left\lvert{0:0{1}b}\right\rangle'.format(0, (len(output_state) - 1).bit_length())
    latex_text = input_state + r'\mathrm{transforms\ to}: ' + format_vector(output_state,decimal_precision=2)
    display(Math(latex_text))

def give_bit(basis, bit):
    if basis == 0:
        bitstring = str(bit)
    elif basis == 1:
        if bit == 0: bitstring = '+'
        elif bit == 1: bitstring = '-'
    else:
        bitstring = ' '
    return bitstring

def noisy_measure(q_circuit, q_reg, c_reg, i_meas = IdGate(label='meas')):
    q_circuit.append(i_meas,[q_reg])
    q_circuit.measure(q_reg, c_reg)
                  
def noisy_swap(q_circuit, q1_reg, q2_reg, noise):
    i_noise = IdGate(label=noise)
    q_circuit.swap(q1_reg, q2_reg)
    q_circuit.append(i_noise,[q1_reg])
    q_circuit.append(i_noise,[q2_reg])
    
def noisy_cnot(q_circuit, q1_reg, q2_reg, noise):
    i_noise = IdGate(label=noise)
    q_circuit.cx(q1_reg, q2_reg)
    q_circuit.append(i_noise,[q1_reg])
    q_circuit.append(i_noise,[q2_reg])
    
# Deze staat nu nog hier voor testen maar dit komt in de QKDfuncties module
def send_photon(q_circuit, c1_reg, e1_reg, c2_reg, e2_reg, p_reg, c_reg, c_classical, theta):
    """ Send a photon from NV center 1 with nuclear spin qubit c1_reg and electron spin e_reg
    to NV center 2 with c2_reg and e_2 reg. Do this via the p_reg with distance L.
    """
    # Sending the entangled photon
    noisy_swap(q_circuit, e1_reg, c1_reg, 'Cswap')                         
    noisy_cnot(q_circuit, e1_reg, p_reg, 'eswap')
    noisy_swap(q_circuit, e1_reg, c1_reg, 'Cswap')           
    q_circuit.barrier(c1_reg, e1_reg, c2_reg, e2_reg, p_reg, c_reg)
    
    # Adding relaxation noise
    q_circuit.ry(theta, c_reg)
    q_circuit.measure(c_reg, c_classical)
    q_circuit.x(c_reg).c_if(c_classical, 1)
    q_circuit.swap(c_reg, p_reg).c_if(c_classical, 1)
    q_circuit.barrier(c1_reg, e1_reg, c2_reg, e2_reg, p_reg, c_reg)
    
    # Receiving the photon
    noisy_swap(q_circuit, e2_reg, p_reg, 'eswap')
    
def send_photon_swap(q_circuit, c1_reg, e1_reg, c2_reg, e2_reg, p_reg, c_reg, c_classical, theta):
    """ Send a photon from NV center 1 with nuclear spin qubit c1_reg and electron spin e_reg
    to NV center 2 with c2_reg and e_2 reg. Do this via the p_reg with distance L.
    """
    # Sending the entangled photon
    noisy_swap(q_circuit, e1_reg, c1_reg, 'Cswap')                         
    noisy_swap(q_circuit, e1_reg, p_reg, 'eswap')        
    q_circuit.barrier(c1_reg, e1_reg, c2_reg, e2_reg, p_reg, c_reg)
    
    # Adding relaxation noise
    q_circuit.ry(theta, c_reg)
    q_circuit.measure(c_reg, c_classical)

    q_circuit.x(c_reg).c_if(c_classical, 1)
    q_circuit.swap(p_reg, c_reg).c_if(c_classical, 1)
    q_circuit.reset(c_reg)
    q_circuit.barrier(c1_reg, e1_reg, c2_reg, e2_reg, p_reg, c_reg)
    
    # Receiving the photon
    noisy_swap(q_circuit, e2_reg, p_reg, 'eswap')


def noisy_x(q_circuit, q_register, noise=None):
    """ Generate a noisy x gate on 'q_register' by applying an extra rotation controlled
    with the parameter 'noise'. 'noise' is the rotation in rad. This parameter and its value 
    should be determined outside of this function. 
    
    If no noise is entered this function generates a perfect gate
    """
    # Check if a noise was entered
    if noise == None:
        q_circuit.x(q_register)
        return
    
    q_circuit.x(q_register)
    q_circuit.rx(noise, q_register)
    
def noisy_ry(theta, q_circuit, q_register, noise=None):
    """ Generate a noisy ry gate with angle 'theta' on 'q_register' by applying a non-perfect y-rotation controlled
    with the parameter 'noise'. This parameter and its value should be determined outside
    of this function. 
    
    If no noise is entered this function generates a perfect gate
    """
    # Check if a noise was entered
    if noise == None:
        q_circuit.ry(theta, q_register)
        return
    
    q_circuit.ry(theta+noise, q_register)
    
def plot_noises(noises, noises_std, labels, titles):
    '''takes a 3D array or list with 2D arrays and plots the data
    noises: different sorts of noise, in (6,5,20) array with (:,0,:) the x-axis
    noises_std: standard deviations of the noises,
    labels: states the noises belong to, in same order,
    titles: names of the noises'''
    fign = plt.figure(figsize=(18,9))
    for k in range(2):
        for j in range(3):
            index = 3*k+j
            ax = fign.add_subplot(2,3,index+1)

            avg_std = np.zeros(20)
            for i, l in enumerate(labels):
                ax.errorbar(noises[index][0,:],noises[index][i+1,:],yerr=noises_std[index][i,:], ls='--',label=l)
                avg_std += (noises_std[index][i,:]/(len(labels)))**2

            avg_std = np.sqrt(avg_std)
                
            ax.errorbar(noises[index][0,:],np.mean(noises[index][1:,:], axis=0), yerr=avg_std, linewidth=3.0,label='avg')
            ax.set_axisbelow(True)
            ax.grid(color='#6b84af', linestyle='dotted')
            plt.ylabel('Quantum Bit Error Rate')
            plt.xlabel(titles[index])
            plt.legend()