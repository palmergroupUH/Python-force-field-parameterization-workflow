module txt_reader
    use system
    implicit none 
    private 

    public :: get_txt_lines,&
              & loadtxt 

contains 
    
    subroutine get_txt_lines(txtfile,strleng,num_lines) bind(c,name="get_txt_lines") 
        implicit none 

        ! Passed: 
        integer(c_int),intent(in) :: strleng 
        character(kind=c_char,len=1),intent(in),dimension(1:strleng) :: txtfile 

        ! Local: 
        character(len=:),allocatable :: filename 
        integer :: IOstatus, unit_number

        ! Output:
        integer(c_int),intent(out) ::  num_lines 

        unit_number = unit_num()
        
        call convert_c_string_f_string(txtfile,strleng,filename) 
       
        open(unit=unit_number,file=filename,action="read",form="formatted") 

            num_lines = 0 

            do 
        
                read(unit_number,*,IOSTAT=IOstatus)

                if ( IOstatus /= 0) then 

                    exit

                end if 

                num_lines = num_lines + 1 

            end do 

        close(unit_number) 

        end subroutine 

    subroutine loadtxt(txtfile,strleng,num_lines,skiprows,loaded_data) bind(c,name="load_txt")
        implicit none 
        integer(c_int),intent(in) :: strleng 
        character(kind=c_char,len=1),dimension(1:strleng),intent(in) :: txtfile
        integer(c_int),intent(in) :: num_lines
        integer(c_int),intent(in) :: skiprows 
        character(len=:),allocatable :: filename 
        integer :: i,unit_number	
        real(c_double),intent(out),dimension(1:num_lines-skiprows) :: loaded_data

        unit_number = unit_num()

        call convert_c_string_f_string(txtfile,strleng,filename) 

        open(unit=unit_number,file=filename,action="read",form="formatted") 

            do i = 1,skiprows
                
                read(unit_number,*)

            end do 

            do i = 1, num_lines-skiprows  

                read(unit_number,*) loaded_data(i) 
                
            end do 

        close(unit_number) 

        end subroutine 


end module 

