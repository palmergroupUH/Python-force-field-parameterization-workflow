program test 
    implicit none 
    integer :: i,unit_num,num_atoms  
    !character(len=:),allocatable :: data_col 
    character(len=1000)  :: atom_type  
    real(8),dimension(1:3,512) :: xyz 

    open(newunit=unit_num,file="results.txt",action="read", form="formatted") 

        read(unit_num,*) num_atoms 
        read(unit_num,*) 

        do i = 1,num_atoms 

            read(unit_num,*) atom_type,xyz(:,i)    
            
        end do  

        print*, xyz 

    close(unit_num) 
end program 
