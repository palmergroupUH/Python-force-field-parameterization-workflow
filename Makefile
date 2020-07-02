SHELL = /bin/sh

all: 

	$(MAKE) --directory=fortranAPI/general  
	$(MAKE) --directory=fortranAPI/IO 
	$(MAKE) --directory=fortranAPI/pair_correlation 
	
	cd fortranAPI/general; $(MAKE) clean  
	cd fortranAPI/IO; $(MAKE) clean  
	cd fortranAPI/pair_correlation; $(MAKE) clean  

