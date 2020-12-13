let SessionLoad = 1
if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
inoremap <C-@> 
inoremap <Nul> 
cnoremap <C-@> <Up>
cnoremap <Nul> <Up>
map  :set cursorline!
noremap  :FZF
noremap  gT|n
noremap  gt|n
noremap  :NERDTreeToggle
noremap  "*
nnoremap  v:NERDTree
vmap  "*d
noremap & zz/;klk;kdkajf
noremap * ma*
xmap , <Plug>Sneak_,
omap , <Plug>Sneak_,
nmap , <Plug>Sneak_,
xmap ; <Plug>Sneak_;
omap ; <Plug>Sneak_;
nmap ; <Plug>Sneak_;
noremap B ^
noremap E $
noremap H 10h
noremap J 10k|zz
noremap K 10j|zz
noremap L 10l
noremap M ma
imap ÎÔ *
map! Îu <C-End>
map! Îw <C-Home>
noremap Q @q
nmap S <Plug>(Scalpel)
nnoremap W 
noremap Z @z
vmap \e <Plug>(ScalpelVisual)
noremap ^ 'a|zz/;klk;kdkajf
vnoremap d d
noremap dd dd
noremap da gg|V|G|d
vmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
nmap gcu <Plug>Commentary<Plug>Commentary
nmap gcc <Plug>CommentaryLine
omap gc <Plug>Commentary
nmap gc <Plug>Commentary
xmap gc <Plug>Commentary
nmap gcp gcap
noremap gg gg|zz
noremap j k|zz
noremap k j|zz
noremap m 'a|zz
xmap s <Plug>Sneak_s
noremap t :tabnew
noremap v  
omap z <Plug>Sneak_s
nnoremap <C-W> v:NERDTree
vnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(netrw#GX(),netrw#CheckIfRemote(netrw#GX()))
nmap <silent> <Plug>CommentaryUndo :echoerr "Change your <Plug>CommentaryUndo map to <Plug>Commentary<Plug>Commentary"
omap <Plug>SneakPrevious <Plug>Sneak_,
omap <Plug>SneakNext <Plug>Sneak_;
xmap <Plug>SneakPrevious <Plug>Sneak_,
xmap <Plug>SneakNext <Plug>Sneak_;
nmap <Plug>SneakPrevious <Plug>Sneak_,
nmap <Plug>SneakNext <Plug>Sneak_;
omap <Plug>(SneakStreakBackward) <Plug>SneakLabel_S
omap <Plug>(SneakStreak) <Plug>SneakLabel_s
xmap <Plug>(SneakStreakBackward) <Plug>SneakLabel_S
xmap <Plug>(SneakStreak) <Plug>SneakLabel_s
nmap <Plug>(SneakStreakBackward) <Plug>SneakLabel_S
nmap <Plug>(SneakStreak) <Plug>SneakLabel_s
xmap <Plug>VSneakPrevious <Plug>Sneak_,
xmap <Plug>VSneakNext <Plug>Sneak_;
xmap <Plug>VSneakBackward <Plug>Sneak_S
xmap <Plug>VSneakForward <Plug>Sneak_s
nmap <Plug>SneakBackward <Plug>Sneak_S
nmap <Plug>SneakForward <Plug>Sneak_s
onoremap <silent> <Plug>SneakLabel_S :call sneak#wrap(v:operator, 2, 1, 2, 2)
onoremap <silent> <Plug>SneakLabel_s :call sneak#wrap(v:operator, 2, 0, 2, 2)
xnoremap <silent> <Plug>SneakLabel_S :call sneak#wrap(visualmode(), 2, 1, 2, 2)
xnoremap <silent> <Plug>SneakLabel_s :call sneak#wrap(visualmode(), 2, 0, 2, 2)
nnoremap <silent> <Plug>SneakLabel_S :call sneak#wrap('', 2, 1, 2, 2)
nnoremap <silent> <Plug>SneakLabel_s :call sneak#wrap('', 2, 0, 2, 2)
onoremap <silent> <Plug>Sneak_T :call sneak#wrap(v:operator, 1, 1, 0, 0)
onoremap <silent> <Plug>Sneak_t :call sneak#wrap(v:operator, 1, 0, 0, 0)
xnoremap <silent> <Plug>Sneak_T :call sneak#wrap(visualmode(), 1, 1, 0, 0)
xnoremap <silent> <Plug>Sneak_t :call sneak#wrap(visualmode(), 1, 0, 0, 0)
nnoremap <silent> <Plug>Sneak_T :call sneak#wrap('', 1, 1, 0, 0)
nnoremap <silent> <Plug>Sneak_t :call sneak#wrap('', 1, 0, 0, 0)
onoremap <silent> <Plug>Sneak_F :call sneak#wrap(v:operator, 1, 1, 1, 0)
onoremap <silent> <Plug>Sneak_f :call sneak#wrap(v:operator, 1, 0, 1, 0)
xnoremap <silent> <Plug>Sneak_F :call sneak#wrap(visualmode(), 1, 1, 1, 0)
xnoremap <silent> <Plug>Sneak_f :call sneak#wrap(visualmode(), 1, 0, 1, 0)
nnoremap <silent> <Plug>Sneak_F :call sneak#wrap('', 1, 1, 1, 0)
nnoremap <silent> <Plug>Sneak_f :call sneak#wrap('', 1, 0, 1, 0)
onoremap <silent> <Plug>SneakRepeat :call sneak#wrap(v:operator, sneak#util#getc(), sneak#util#getc(), sneak#util#getc(), sneak#util#getc())
onoremap <silent> <Plug>Sneak_S :call sneak#wrap(v:operator, 2, 1, 2, 1)
onoremap <silent> <Plug>Sneak_s :call sneak#wrap(v:operator, 2, 0, 2, 1)
xnoremap <silent> <Plug>Sneak_S :call sneak#wrap(visualmode(), 2, 1, 2, 1)
xnoremap <silent> <Plug>Sneak_s :call sneak#wrap(visualmode(), 2, 0, 2, 1)
nnoremap <silent> <Plug>Sneak_S :call sneak#wrap('', 2, 1, 2, 1)
nnoremap <silent> <Plug>Sneak_s :call sneak#wrap('', 2, 0, 2, 1)
nnoremap <Plug>(Scalpel) :Scalpel/\v<=escape(expand('<cword>'), "!\"#$%&'()*+,-.\/:;<=>?@[\\]^`{|}~")>//<Left>
nmap <F5> ysl
nmap <F4> yse
nmap <F2> <Plug>Sneak_S
nmap <F1> <Plug>Sneak_s
noremap <C-T> :NERDTreeToggle
noremap <F9> :term py %
noremap <F8> :TagbarToggle
noremap <C-F> :FZF
noremap <C-H> gT|n
noremap <C-L> gt|n
noremap <C-V> "*
map <C-C> :set cursorline!
inoremap 	 
vmap ÎØ "*d
vmap Î× "*d
vmap ÎÕ "*y
vmap ÎÔ "-d"*P
nmap ÎÔ "*P
nmap Îu <C-End>
vmap Îu <C-End>
nmap Îw <C-Home>
vmap Îw <C-Home>
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set backupdir=~\\Documents\\vim_backup
set completeopt=longest,menuone
set expandtab
set helplang=En
set ignorecase
set laststatus=2
set runtimepath=~/vimfiles,~\\.vim\\plugged\\awesome-vim-colorschemes,~\\.vim\\plugged\\nerdtree,~\\.vim\\plugged\\scalpel,~\\.vim\\plugged\\vim-rainbow,~\\.vim\\plugged\\lightline.vim,~\\.vim\\plugged\\fzf,~\\.vim\\plugged\\tagbar,~\\.vim\\plugged\\vim-numbertoggle,~\\.vim\\plugged\\vim-sneak,~\\.vim\\plugged\\vim-commentary,C:\\Program\ Files\ (x86)\\Vim/vimfiles,C:\\Program\ Files\ (x86)\\Vim\\vim82,C:\\Program\ Files\ (x86)\\Vim/vimfiles/after,~\\.vim\\plugged\\awesome-vim-colorschemes\\after,~/vimfiles/after
set shiftwidth=4
set noshowmode
set tabline=%!lightline#tabline()
set tabstop=4
set termguicolors
set wildignore=*.pyc
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd E:\School\Winter_2020_2021\RNF_Kovar\RNF-proj
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
argglobal
%argdel
set stal=2
tabnew
tabnew
tabrewind
edit neural_network.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),0],:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal completeslash=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal cursorlineopt=both
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(substitute(substitute(v:fname,b:grandparent_match,b:grandparent_sub,''),b:parent_match,b:parent_sub,''),b:child_match,b:child_sub,'g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=0{,0},0),0],:,!^F,o,O,e,<:>,=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=python\ -m\ pydoc\ 
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal scrolloff=-1
setlocal shiftwidth=4
setlocal noshortname
setlocal showbreak=
setlocal sidescrolloff=-1
setlocal signcolumn=auto
setlocal nosmartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=%{lightline#link()}%#LightlineLeft_active_0#%(\ %{lightline#mode()}\ %)%{(&paste)?\"|\":\"\"}%(\ %{&paste?\"PASTE\":\"\"}\ %)%#LightlineLeft_active_0_1#%#LightlineLeft_active_1#%(\ %R\ %)%{(&readonly)&&(LightlineFilename()!=#\"\"||(&modified||!&modifiable))?\"|\":\"\"}%(\ %{LightlineFilename()}\ %)%{LightlineFilename()!=#\"\"&&((&modified||!&modifiable))?\"|\":\"\"}%(\ %M\ %)%#LightlineLeft_active_1_2#%#LightlineMiddle_active#%=%#LightlineRight_active_2_3#%#LightlineRight_active_2#%(\ %{&ff}\ %)%{1||1?\"|\":\"\"}%(\ %{&fenc!=#\"\"?&fenc:&enc}\ %)%{1?\"|\":\"\"}%(\ %{&ft!=#\"\"?&ft:\"no\ ft\"}\ %)%#LightlineRight_active_1_2#%#LightlineRight_active_1#%(\ %3p%%\ %)%#LightlineRight_active_0_1#%#LightlineRight_active_0#%(\ %3l:%-2c\ %)
setlocal suffixesadd=.py
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=8
setlocal tagcase=
setlocal tagfunc=
setlocal tags=
setlocal termwinkey=
setlocal termwinscroll=10000
set termwinsize=60x400
setlocal termwinsize=60x400
setlocal textwidth=0
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal varsofttabstop=
setlocal vartabstop=
setlocal wincolor=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 11 - ((10 * winheight(0) + 30) / 61)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
11
normal! 017|
tabnext
edit simple_example.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),0],:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal completeslash=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal cursorlineopt=both
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(substitute(substitute(v:fname,b:grandparent_match,b:grandparent_sub,''),b:parent_match,b:parent_sub,''),b:child_match,b:child_sub,'g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=0{,0},0),0],:,!^F,o,O,e,<:>,=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=python\ -m\ pydoc\ 
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
set relativenumber
setlocal relativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal scrolloff=-1
setlocal shiftwidth=4
setlocal noshortname
setlocal showbreak=
setlocal sidescrolloff=-1
setlocal signcolumn=auto
setlocal nosmartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=%{lightline#link()}%#LightlineLeft_active_0#%(\ %{lightline#mode()}\ %)%{(&paste)?\"|\":\"\"}%(\ %{&paste?\"PASTE\":\"\"}\ %)%#LightlineLeft_active_0_1#%#LightlineLeft_active_1#%(\ %R\ %)%{(&readonly)&&(LightlineFilename()!=#\"\"||(&modified||!&modifiable))?\"|\":\"\"}%(\ %{LightlineFilename()}\ %)%{LightlineFilename()!=#\"\"&&((&modified||!&modifiable))?\"|\":\"\"}%(\ %M\ %)%#LightlineLeft_active_1_2#%#LightlineMiddle_active#%=%#LightlineRight_active_2_3#%#LightlineRight_active_2#%(\ %{&ff}\ %)%{1||1?\"|\":\"\"}%(\ %{&fenc!=#\"\"?&fenc:&enc}\ %)%{1?\"|\":\"\"}%(\ %{&ft!=#\"\"?&ft:\"no\ ft\"}\ %)%#LightlineRight_active_1_2#%#LightlineRight_active_1#%(\ %3p%%\ %)%#LightlineRight_active_0_1#%#LightlineRight_active_0#%(\ %3l:%-2c\ %)
setlocal suffixesadd=.py
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=8
setlocal tagcase=
setlocal tagfunc=
setlocal tags=
setlocal termwinkey=
setlocal termwinscroll=10000
set termwinsize=60x400
setlocal termwinsize=60x400
setlocal textwidth=0
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal varsofttabstop=
setlocal vartabstop=
setlocal wincolor=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 43 - ((30 * winheight(0) + 30) / 61)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
43
normal! 020|
tabnext
edit original_example.py
set splitbelow splitright
set nosplitbelow
set nosplitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal balloonexpr=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),0],:,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=b:#,fb:-
setlocal commentstring=#\ %s
setlocal complete=.,w,b,u,t,i
setlocal concealcursor=
setlocal conceallevel=0
setlocal completefunc=
setlocal completeslash=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
setlocal nocursorline
setlocal cursorlineopt=both
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal expandtab
if &filetype != 'python'
setlocal filetype=python
endif
setlocal fixendofline
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=tcq
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal formatprg=
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=-1
setlocal include=^\\s*\\(from\\|import\\)
setlocal includeexpr=substitute(substitute(substitute(v:fname,b:grandparent_match,b:grandparent_sub,''),b:parent_match,b:parent_sub,''),b:child_match,b:child_sub,'g')
setlocal indentexpr=GetPythonIndent(v:lnum)
setlocal indentkeys=0{,0},0),0],:,!^F,o,O,e,<:>,=elif,=except
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=python\ -m\ pydoc\ 
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeencoding=
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=bin,octal,hex
set number
setlocal number
setlocal numberwidth=4
setlocal omnifunc=
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal scrolloff=-1
setlocal shiftwidth=4
setlocal noshortname
setlocal showbreak=
setlocal sidescrolloff=-1
setlocal signcolumn=auto
setlocal nosmartindent
setlocal softtabstop=4
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=%{lightline#link()}%#LightlineLeft_active_0#%(\ %{lightline#mode()}\ %)%{(&paste)?\"|\":\"\"}%(\ %{&paste?\"PASTE\":\"\"}\ %)%#LightlineLeft_active_0_1#%#LightlineLeft_active_1#%(\ %R\ %)%{(&readonly)&&(LightlineFilename()!=#\"\"||(&modified||!&modifiable))?\"|\":\"\"}%(\ %{LightlineFilename()}\ %)%{LightlineFilename()!=#\"\"&&((&modified||!&modifiable))?\"|\":\"\"}%(\ %M\ %)%#LightlineLeft_active_1_2#%#LightlineMiddle_active#%=%#LightlineRight_active_2_3#%#LightlineRight_active_2#%(\ %{&ff}\ %)%{1||1?\"|\":\"\"}%(\ %{&fenc!=#\"\"?&fenc:&enc}\ %)%{1?\"|\":\"\"}%(\ %{&ft!=#\"\"?&ft:\"no\ ft\"}\ %)%#LightlineRight_active_1_2#%#LightlineRight_active_1#%(\ %3p%%\ %)%#LightlineRight_active_0_1#%#LightlineRight_active_0#%(\ %3l:%-2c\ %)
setlocal suffixesadd=.py
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'python'
setlocal syntax=python
endif
setlocal tabstop=8
setlocal tagcase=
setlocal tagfunc=
setlocal tags=
setlocal termwinkey=
setlocal termwinscroll=10000
set termwinsize=60x400
setlocal termwinsize=60x400
setlocal textwidth=0
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal varsofttabstop=
setlocal vartabstop=
setlocal wincolor=
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 3 - ((2 * winheight(0) + 30) / 61)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
3
normal! 0
tabnext 2
set stal=1
badd +0 neural_network.py
badd +0 simple_example.py
badd +0 original_example.py
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToOS
set winminheight=1 winminwidth=1
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
