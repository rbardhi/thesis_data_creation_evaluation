from create_input_program_hier import create_input_program, create_input_program_det, create_input_program_noisy
  
def write_dc_input_det(examples,constrained,twoD,filename):
  dc_file = create_input_program(examples,constrained,twoD)
  with open(filename,'w') as f:
    f.write(dc_file)
def write_dc_input_noisy(examples,constrained,twoD,filename_det, filename_noisy, noise_type, noise_val):
  dc_file_det = create_input_program_det(examples,constrained,twoD, noise_type) #done
  dc_file_noisy = create_input_program_noisy(examples,constrained,twoD, noise_type, noise_val)
  with open(filename_det,'w') as f:
    f.write(dc_file_det)
  with open(filename_noisy,'w') as f:
    f.write(dc_file_noisy)
  
