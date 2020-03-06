!--------------------------------------------------------------------------------
!---------------------------- Parallel Initialize -------------------------------
!--------------------------------------------------------------------------------

subroutine assign_workload_to_workers(cores_given,total_configs,workload_perworker) 
	implicit none 
	integer,intent(in) :: cores_given,total_configs    
	integer :: num_config_percore, ncores,remainder 
	integer,intent(out),dimension(1:cores_given)  :: workload_perworker 

	num_config_percore = total_configs/cores_given

	do ncores = 1,cores_given 

		workload_perworker(ncores) = num_config_percore 

	end do  

	remainder  = mod(total_configs,cores_given) 

	workload_perworker(cores_given) = workload_perworker(cores_given) + remainder 

	end subroutine 

subroutine assign_task_to_workers(cores_given,workload,read_start) 
	implicit none 
	integer,intent(in) :: cores_given
	integer,intent(in),dimension(:) ::  workload 
	integer,intent(out),dimension(1:cores_given)  :: read_start
	integer :: core,read_end 

	! initialize first element

	read_start(1) = 1 ; read_end = workload(1) 

	do core =  2,cores_given  

		read_start(core) = read_end + 1 

		read_end = read_start(core) + workload(core ) -1  

	end do 	

	end subroutine 

subroutine load_work_based_on_buffersize(buffersize,workload,load_times,loaded_work )
	implicit none 
	integer,intent(in) :: buffersize,workload,load_times 
	integer,intent(out),dimension(1:load_times) :: loaded_work  

	if ( workload <= buffersize ) then 

		loaded_work(load_times) = workload 	

	else 
	
		loaded_work = buffersize 

		loaded_work(load_times) = mod(workload,buffersize) 

	end if 

	end subroutine  

!--------------------------------------------------------------------------------
!------------------------------- Load TXT data ---------------------------------
!--------------------------------------------------------------------------------

subroutine getlines(filename,num_lines) 
	implicit none 
	character(len=*),intent(in) :: filename 

	integer :: IOstatus
	integer,intent(out) ::  num_lines 

	open(unit=1991,file=filename,action="read",form="formatted") 

		num_lines = 0 

		do 
	
			read(1991,*,IOSTAT=IOstatus)

			if ( IOstatus /= 0) then 

				exit

			end if 

			num_lines = num_lines + 1 

		end do 

	close(1991) 

	end subroutine 

subroutine loadtxt(filename, num_lines,skiprows,loaded_data)
	implicit none 
	character(len=*),intent(in) :: filename 
	integer,intent(in) :: num_lines
	integer,intent(in) :: skiprows 
	integer :: i		
	real(8),intent(out),dimension(1:num_lines-skiprows) :: loaded_data

	open(unit=1991,file=filename,action="read",form="formatted") 

		do i = 1,skiprows
			
			read(1991,*)

		end do 

		do i = 1, num_lines-skiprows  

			read(1991,*) loaded_data(i) 
			
		end do 

	close(1991) 

	end subroutine 

!--------------------------------------------------------------------------------
!------------------------------- Load DCD data ---------------------------------
!--------------------------------------------------------------------------------

subroutine readdcdheader(dcdfile,total_atoms,total_frames) 
	implicit none 
	character(len=*),intent(in) :: dcdfile
	integer,dimension(20) :: ICNTRL
	character(len=3) :: fext
	character(len=4) :: HDR
	real(4) :: delta
	integer :: i,unitnum
	integer,intent(out) :: total_atoms,total_frames 

	unitnum = 123 

	open(unit=unitnum,file=trim(dcdfile),form='unformatted',status='old')

		read(unitnum) HDR, (ICNTRL(i),i=1,9),delta,(ICNTRL(i),i=11,20)
		read(unitnum)
		read(unitnum) total_atoms 

		total_frames =  ICNTRL(1)

	close(unitnum)

	end subroutine 
	
subroutine read_DCD_xyzbox(dcdfile,total_atoms,current_frame,xyz,box) 
	implicit none 
	character(len=150),intent(in)  :: dcdfile 
	integer,intent(in) :: current_frame,total_atoms
	integer ::i,iframe,imol,counter,unitnum 
	real(8),dimension(6) :: XTLABC 
	real(8),intent(out),dimension(1:3) :: box 
	real(4),intent(out),dimension(1:3,total_atoms) :: xyz  

	unitnum = 123 

	open(unit=unitnum,file=trim(dcdfile),form='unformatted',status='old') 

		read(unitnum)
		read(unitnum)
		read(unitnum)

		!counter = 0 

		do i = 1, current_frame - 1  

			!counter = counter + 1 

			read(unitnum)
			read(unitnum)
			read(unitnum)
			read(unitnum)

		end do 

		!counter = counter + 1 
		
		!if ( counter == current_frame) then  	

		read(unitnum) (XTLABC(i),i=1,6) 
		read(unitnum) (xyz(1,imol),imol=1,total_atoms) 
		read(unitnum) (xyz(2,imol),imol=1,total_atoms) 
		read(unitnum) (xyz(3,imol),imol=1,total_atoms) 

		box =  [ XTLABC(1),XTLABC(3),XTLABC(6) ] 

		!end if 

	close(unitnum) 

	end subroutine 

subroutine read_dcd_in_chunk(start_at,num_configs,dcdfile,total_atoms,xyz,box) 
	implicit none 
	character(len=150),intent(in)  :: dcdfile 
	integer,intent(in) :: start_at,total_atoms,num_configs
	integer ::i,iframe,imol,counter,unitnum 
	real(8),dimension(6) :: XTLABC 
	
	real(8),intent(out),dimension(1:3,num_configs) :: box 	
	real(4),intent(out),dimension(1:3,total_atoms,num_configs) :: xyz 	

	unitnum = 10000 

	counter = 0 

	open(unit=unitnum,file=trim(dcdfile),form='unformatted',status='old') 
	
		! DCD Header 

		read(unitnum)
		read(unitnum)
		read(unitnum)

		! Skip these frames 

		do i = 1, start_at - 1 		

			read(unitnum)
			read(unitnum)
			read(unitnum)
			read(unitnum) 

		end do 
	
		! Read these frames 	
		
		do iframe = 1,num_configs

			read(unitnum) (XTLABC(i),i=1,6)
			read(unitnum) xyz(1,:,iframe)  
			read(unitnum) xyz(2,:,iframe) 
			read(unitnum) xyz(3,:,iframe)

			box(1:3,iframe) =  [ XTLABC(1),XTLABC(3),XTLABC(6) ] 

		end do 

	close(unitnum) 	

	end subroutine   

! Read only the volume in dcd trajectory 
subroutine read_volume_dcd_in_chunk(dcdfile,start_at,num_configs,box)    
	implicit none 

	! Passed 
	character(len=150),intent(in)  :: dcdfile 
	integer,intent(in) :: start_at,num_configs

	! Local 
	integer ::i,iframe,imol,counter,unitnum 
	real(8),dimension(6) :: XTLABC 
	
	! Return 
	real(8),intent(out),dimension(1:3,num_configs) :: box 	

	unitnum = 10000 

	counter = 0 

	open(unit=unitnum,file=trim(dcdfile),form='unformatted',status='old') 
	
		! DCD Header 

		read(unitnum)
		read(unitnum)
		read(unitnum)

		! Skip these frames 

		do i = 1, start_at - 1 		

			read(unitnum)
			read(unitnum)
			read(unitnum)
			read(unitnum) 

		end do 
	
		! Read these frames 	
		
		do iframe = 1,num_configs

			read(unitnum) (XTLABC(i),i=1,6)
			read(unitnum) 
			read(unitnum) 
			read(unitnum) 

			box(1:3,iframe) =  [ XTLABC(1),XTLABC(3),XTLABC(6) ] 

		end do 

	close(unitnum) 	

	end subroutine 

! Read center of mass for a group of atoms
subroutine read_com_dcd_in_chunk(start_at,num_configs,dcdfile,total_atoms,mass_list,nsites,xyz_cm,box )
	implicit none 
	! Passed 
	character(len=150),intent(in)  :: dcdfile 
	integer,intent(in) :: start_at,total_atoms,num_configs,nsites
	real(4),intent(in),dimension(:) :: mass_list 
	! Local 
	integer ::i,iframe,imol,counter,unitnum,num_mol,start,last 
	real(4) :: total_mass 
	real(4),dimension(1:3,total_atoms) :: xyz_all	
	real(8),dimension(6) :: XTLABC 

	! Return 	
	real(8),intent(out),dimension(1:3,num_configs) :: box 	
	real(4),intent(out),dimension(1:3,total_atoms/nsites,num_configs) :: xyz_cm 	

	unitnum = 10000 

	counter = 0 

	open(unit=unitnum,file=trim(dcdfile),form='unformatted',status='old') 
	
		! DCD Header 

		read(unitnum)
		read(unitnum)
		read(unitnum)

		! Skip these frames 

		do i = 1, start_at - 1 		

			read(unitnum)
			read(unitnum)
			read(unitnum)
			read(unitnum) 

		end do 
	
		! Read these frames 
		
		num_mol = total_atoms/nsites 

		total_mass = sum(mass_list) 

		do iframe = 1,num_configs

			read(unitnum) (XTLABC(i),i=1,6)
			read(unitnum) xyz_all(1,:)  
			read(unitnum) xyz_all(2,:) 
			read(unitnum) xyz_all(3,:)

			box(1:3,iframe) =  [ XTLABC(1),XTLABC(3),XTLABC(6) ] 

			do imol = 1, num_mol

				start = (imol-1)*nsites + 1  

				last = start + nsites - 1  

				xyz_cm(1,imol,iframe) = sum(xyz_all(1,start:last)*mass_list)/total_mass 

				xyz_cm(2,imol,iframe) = sum(xyz_all(2,start:last)*mass_list)/total_mass 

				xyz_cm(3,imol,iframe) = sum(xyz_all(3,start:last)*mass_list)/total_mass 
	
			end do 
	
		end do 

	close(unitnum) 	

	end subroutine 

!--------------------------------------------------------------------------------
!------------------------- Load text data by bytes ------------------------------
!--------------------------------------------------------------------------------

subroutine convert_lines_to_bytes(filename,start_lines,line_in_bytes)
	! Divide the whole files into equal chunk marked by bytes   
	! Then use the marked position to read the chunk of data 

	! Input: 
	! filename: text files name to be read
	! nlines: numbers of lines for each chunk  

	!Return 
	! start_lines ( integer ) : the next chunk positions ( in bytes )  

	implicit none
	integer,intent(in) :: start_lines 
	character(len=*), intent(in) :: filename
	integer :: i, int1,int2
	integer,intent(out) :: line_in_bytes

	open(unit=222,file=filename,action="read",form="formatted",status="old",access="stream")

		do i = 1,start_lines - 1 

			read(222,*)

		end do

		inquire(unit=222,pos = line_in_bytes )

	close(222)

	end subroutine

subroutine jump_lammpsoutput_in_bytes(filename,start_at_bytes,natoms,num_configs,end_bytes)  
	implicit none 
	integer,intent(in):: start_at_bytes,num_configs,natoms  
	character(len=*), intent(in) :: filename 
	integer :: i 
	integer, intent(out) :: end_bytes 
	
	open(unit=222,file=filename,action="read",form="formatted",status="old",access="stream")

		read(unit=222,pos=start_at_bytes) 	
	
		do i = 1, num_configs*(natoms+9) 

			read(222,*) 	

		end do 

		inquire(unit=222,pos=end_bytes) 	
	
	end subroutine 

subroutine jump_lammpsoutput_lines(natoms,start_configs,jump_lines) 
	implicit none 
	integer,intent(in) :: natoms,start_configs 
	integer,intent(out) :: jump_lines	

	if ( start_configs == 1 ) then  

		jump_lines = 1

	else 

		jump_lines = (start_configs-1)*(natoms+9) 	

	end if 

	end subroutine 

subroutine read_lammps_force_output(filename,natoms,startpos,nconfigs,force) 
	implicit none 
	character(len=*) ,intent(in) :: filename 
	integer,intent(in) :: nconfigs, startpos,natoms 
	integer :: counter,loc,i, int1,int2,firstline,configcounter,total_lines  
	real(8), intent(out),dimension(natoms*nconfigs*3) :: force 

	open(unit=1991,file=filename,action="read",form="formatted",status="old",access="stream") 

		counter = 1 ; loc = 0 ; firstline = 0 ; configcounter = 0  

		if ( startpos == 1 ) firstline = 1  

		read(1991,*,pos=startpos) 	

		do while ( counter <=  nconfigs  )   

			counter = counter + 1 

			do i = 1, 9 - firstline

				read(1991,*) 
			
			end do 
	
			do i = 1,natoms

				loc = 3*(i-1) + natoms*configcounter*3 + 1  
	
				read(1991,*) int1,int2,force(loc:loc+2) 

			end do 
		
			configcounter = configcounter + 1 
	
			firstline = 0 

		end do 

	close(1991) 

	end subroutine 


