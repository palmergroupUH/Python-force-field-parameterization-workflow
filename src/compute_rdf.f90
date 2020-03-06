subroutine build_homo_pair_distance_histogram(total_atoms,cutoff,num_bins,XYZ,L_xyz,rdf_histogram)  
	implicit none
	
	! Passed 
	integer,intent(in) :: total_atoms
	real(8),intent(in) :: cutoff,num_bins
	real(8),intent(in),dimension(:,:) :: XYZ
	real(8),intent(in),dimension(:) :: L_xyz

	! Local 
	real(8),dimension(1:size(L_xyz)) :: xyz_temp,xyz_separate
	integer :: i,j,k,N_ATOM,atom_i,atom_j,index_s,index_e,bin_index,counter
	real(8) :: bin_incre,ave_density,Distance_sqr,Distance,cutoff_sqr,r_interval 

	! Output 
	real(8),intent(out),dimension(1:num_bins) :: rdf_histogram

	r_interval = cutoff/num_bins 

	cutoff_sqr = cutoff*cutoff

	rdf_histogram = 0.0d0 

	do i = 1, total_atoms-1

		xyz_temp = xyz(1:3,i)

		do j = i + 1,total_atoms

			! FIND SEPARATION VECTOR

			xyz_separate = xyz_temp - xyz(1:3,j)

			! APPLY PERIODICAL BOUNDARY CONDITION

			xyz_separate = xyz_separate - L_xyz*dnint(xyz_separate/L_xyz)

			! FIND THE SUM SQUARE OF SEPARATION VECTOR

			Distance_sqr = xyz_separate(1)*xyz_separate(1) + xyz_separate(2)*xyz_separate(2) + xyz_separate(3)*xyz_separate(3)

			if (Distance_sqr < cutoff_sqr ) then 

				bin_index = int(dsqrt(Distance_sqr)/r_interval) + 1

				rdf_histogram(bin_index)  =  rdf_histogram(bin_index) + 2

			end if

		end do 

	end do

	end subroutine 

subroutine accumulate(vol,natoms,volume,total_atoms)
	implicit none 

	! Passed 
	real(8),intent(in) :: vol 	
	integer,intent(in) :: natoms

	! Local 

	! Output 
	real(8),intent(inout) :: volume,total_atoms 

	volume = volume + vol 
	
	total_atoms = total_atoms + natoms 	

	end subroutine 

real(8) function bulk_density(total_atoms,total_volume) 
	implicit none 
	integer,intent(in) :: total_atoms
	real(8),intent(in) :: total_volume 

	bulk_density = total_atoms/total_volume 
		
	end function  

subroutine normalize_histogram(rdf_histogram,num_bins,cutoff,natoms,num_configs,bulk_density,gr)
	implicit none  

	! Passed 
	integer,intent(in) :: num_bins,natoms,num_configs
	real(8),intent(in),dimension(:) :: rdf_histogram 
	real(8),intent(in) :: cutoff,bulk_density  

	! Local
	real(8) :: preconst,vshell_i,half_interval,upper_r,lower_r,center_r,r_interval  
	integer :: i 

	! Output

	real(8),dimension(1:num_bins),intent(out) :: gr  

	preconst = 4.0d0/3.0d0*3.14159265358979d0

	r_interval = cutoff/num_bins

	half_interval = r_interval/2.0d0

	do i = 1,num_bins 

		upper_r = i*r_interval 

		lower_r = (i-1)*r_interval

		center_r = half_interval + r_interval*(i-1)

		vshell_i = preconst*( ( upper_r*upper_r*upper_r ) - (lower_r*lower_r*lower_r) )  

		gr(i) = rdf_histogram(i)/( vshell_i*bulk_density*natoms*num_configs)

	end do 				

	end subroutine 

subroutine calc_coordination_number(gr,r_interval,r_dist,r_lower,r_upper,density,cn)
	implicit none 
	! Passed 
	real(8),intent(in),dimension(:) :: gr,r_dist 
	real(8),intent(in) :: r_lower,r_upper,density,r_interval  
	! Local 

	integer :: i, lower_index, upper_index
	real(8) :: sum_gr,const 
	! Return 
	
	real(8),intent(out) :: cn  

	lower_index = int(r_lower/r_interval) + 1 

	upper_index = int(r_upper/r_interval) + 1  
	
	sum_gr = 0.0d0  

	const = 4*3.14159265358979d0 

	do i = 1,lower_index, upper_index 

		sum_gr = sum_gr + (r_dist(i)**2* gr(i) )*r_interval*density  	

	end do 
	
	cn = sum_gr*const 
	
	end subroutine
	

subroutine write_pcf()  
	implicit none 



 	end subroutine  



