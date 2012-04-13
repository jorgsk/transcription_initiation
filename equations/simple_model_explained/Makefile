NAME=short_simple
CLEAN_FILES = *.log *.aux *.toc *.bbl *.blg ${NAME}.pdf *.xml *.bcf

RM = rm -f

default : pdf

$(NAME).pdf : $(NAME).tex
	xelatex $(NAME)
	@if(grep "There were undefined references" $(NAME).log > /dev/null);\
	then \
		biber $(NAME); \
		xelatex $(NAME); \
	fi
	
	@if(grep "Rerun" $(NAME).log > /dev/null);\
	then \
		 xelatex $(NAME);\
	fi
	rm -f $(NAME).log

pdf : $(NAME).pdf

.PHONY : clean
clean :
	$(RM) ${CLEAN_FILES}


