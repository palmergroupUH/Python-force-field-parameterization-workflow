def is_int(a):

    try: 
    
        int(a)  

        return True 

    except ValueError:

        return False

def is_float(a):

    try:

        float(a)
    
        #return True 

    except ValueError:

        return False

    else: 

        return True

def is_string(a):

    if ( not is_float(a) and not is_int(a) ):

        return True

    else:

        return False

def contain_no_number(contents_lst):

    not_number = 0 

    for i in range(len(contents_lst)):
    
        if ( not is_float(contents_lst[i]) and not is_int(contents_lst[i]) ):

            not_number +=1  

    if ( not_number == len(contents_lst)):

        return True 

    else:
    
        return False

