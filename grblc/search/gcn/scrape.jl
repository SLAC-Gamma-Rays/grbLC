using HTTP , CSV, DataFrames
function doanalysis()
    dfg=nothing
    function change(df)
    	name=names(df)
   		listTime = ["time", "Tmid", "T0", "t-T0", "t-to","Tmid-T0  ","t-T0,d"]
    	listMag = ["mag", "Magnitude", "OT", "magn","magnitude","mag,","Mag.","mag."]
    	listLimit=["UL(3sigma)","UL(3 sigma)","limit","lim","UL."]
		listDate=["Date","date"]
		listExp=["Exp.","Exp(s)","Exps"," Expt. ","Exptime"]
		listMagerr=["Err","Err.","Mag_err"]
		listFilter=["Filt.","filter"]
		listTele=["          Site       "]
    	for i in 1:length(name)
        	if name[i] in listTime; rename!(df,Dict(name[i]=>"Time"))
				elseif name[i] in listMag; rename!(df,Dict(name[i]=>"Mag"))
				elseif name[i] in listLimit; rename!(df,Dict(name[i]=>" Limit"))
				elseif name[i] in listDate; rename!(df,Dict(name[i]=>"      Date Time      "))
				elseif name[i] in listExp; rename!(df,Dict(name[i]=>"Exp"))
				elseif name[i] in listMagerr; rename!(df,Dict(name[i]=>"MagErr"))
				elseif name[i] in listFilter; rename!(df,Dict(name[i]=>"Filter"))
				elseif name[i] in listTele; rename!(df,Dict(name[i]=>"Telescope"))
			end
    			i=+1
    	end
	end
	
       for x in 1:36000
           print("\r peeking at GCN $x ")
           try
               #url = "https://gcn.nasa.gov/circulars/$x.txt"
               #resp = HTTP.get(url) 
               #status=resp.status
               #print(" ",status," "); 
               #if status == 404 ; println("status=",status); continue; end          
               #txt = String(resp.body)
                txt=read("/home/raman/Documents/archive/archive.txt/$x.txt", String)
                grb_rexp=r"GRB ?\d{6}([A-G]|(\.\d{2}))?"
				m=match(grb_rexp,txt)
				grb="nogrb"
                if occursin(grb_rexp,txt)
				    print(m.match)
					grb=m.match
			    end
			    function dframe(df)
					df.GCN=[x for i in 1:nrow(df)]
    				df.GRB=[grb for i in 1:nrow(df)]
    				if isnothing(dfg) 
    					dfg=df
    				else
        				dfg=vcat(dfg,df,cols=:union)
        			end # if x is first
    			end  # dframe
			  
                if occursin("V. Lipunov", txt)
                    println(" MASTER report")                                
                    he=first(findfirst(r"^Tmid"im,txt))
                    lr=first(findnext("\nFilter",txt,he))-1
                    cltxt=txt[he:lr]                                 
                    df=CSV.read(IOBuffer(cltxt), DataFrame, delim='|', skipto=3)
                    if " Limit" in names(df); rename!(df, " Limit" => "Mag"); end
					df.System=["Vega" for i in 1:nrow(df)]
					df.GalExt=["n" for i in 1:nrow(df)]
					change(df)
					
					
                elseif occursin("report on behalf of the Swift/UVOT team",txt)
                	println(" SWIFT/UVOT report")
            		hb,he=findfirst(r"^Filter"im,txt)
					lr,_=findnext("\n\nThe",txt,he)
					cltxt=replace(txt[hb:lr], r"   +"=>s"|",r" ?\+/?- ?"=>s"|",">"=>"",r"(\s\n)"=>s"\n", "�"=>"|")
					ltxt=txt[lr:end]
					gal=match(r"(\s+are\s+(not)\s+corrected\s+for\s+(the\s+)?Galactic\s+extinction\s+)",ltxt)
					gext=gal.captures[2]
					if gext==nothing
						galext="y"
					elseif gext=="not"
						galext="n"
					end
					df=CSV.read(IOBuffer(cltxt), DataFrame, delim='|')
					if "Column6" in names(df); rename!(df, :Column6 => :MagErr); end
					df.Time = (df."T_start(s)" .+ df."T_stop(s)") ./ 2
					df.GalExt=[galext for i in 1:nrow(df)]
					df.Telescope=["SWIFT/UVOT" for i in 1:nrow(df)]
					change(df)
			        
			        
			    elseif occursin("GROND", txt)
                	println(" GROND report") 
                	he=first(findfirst(r"^(( *)?)(g|r|i|z|J|H|K)'?[^\.\w(g'r'i'z'JHK)]"m,txt))
			    	ptxt=txt[1:he]
					t=match(r"((\d+\.?)(\d{1,2})?)( {1,2})?-?(\b((minutes)|(mins?)|m\s)|((seconds?)|(sec)|s\s)|((hours?)|(hr?s?))|(days?)\b)(\s+)(after(\s+\w*)+trigger)",ptxt)
					if t==nothing
						continue
					end
					tar=tryparse(Float32,t.captures[1])
					if t.captures[15]!=nothing
						ti=tar*86400
					elseif t.captures[12] != nothing
						ti=tar*3600
					elseif t.captures[6] !=nothing
						ti=tar*60
					else ti=tar
					end
				
					ep=collect(eachmatch(r"((\d+\.?)(\d{1,2})?)( {1,2})?-?(\b((minutes)|(mins?)|m\s)|((seconds?)|(sec)|s\s)|((hours?)|(hr?s?))\b)",ptxt))
					if ep==RegexMatch[]
						continue
					else e=last(ep)
					end
					par=tryparse(Float32,e.captures[1])
					if e.captures[12] != nothing
						p=par*3600
					elseif e.captures[6] !=nothing
						p=par*60
					else p=par
					end
					tim=ti+(p/2)
					lr=first(findnext(r"^(((?:[\t ]*(?:\r?\n|\r))+)|(The)|(Given)|(This))"m,txt,he))-1
					ltxt=txt[lr:end]
					ys=(collect(eachmatch(r"((Vega)|(Johnson)|(AB)|(SDSS)|(DSS)|(2MASS)|(USNO)|(NOMAD)|(GSC)|(ST))",ptxt)))
					if ys==RegexMatch[]
						continue
					else sys=last(ys)
					end
					if sys==nothing
						sy=match(r"((Vega)|(Johnson)|(AB)|(SDSS)|(DSS)|(2MASS)|(USNO)|(NOMAD)|(GSC)|(ST))",ltxt)
						syst=sy.captures[1]
					else syst=sys.captures[1]
					end
					ga=match(r"(\s+are\s+(not)\s+corrected\s+for\s+((\w*)?(\s+)?)*?Galactic\s+(\w*)?(\s+)?(\3)?extinction\s+)",ltxt)
					if ga==nothing
						galex="n"
					elseif ga!=nothing
						gal=ga.captures[2]
						if gal==nothing
							galex="y"
						else galex="n"
						end
					end
                	cltxt=replace(txt[he:lr], "mag"=>"",","=>" ",r" ?(=|>|~|<)"=>"|" , r"\+/?-"=>"|","�"=>" ","and"=>"")
                	df=CSV.read(IOBuffer(cltxt), DataFrame, delim="|" ,header=0)
					df.Time=[tim for i in 1:nrow(df)]
					df.Exp=[p for i in 1:nrow(df)]
					df.System=[syst for i in 1:nrow(df)]
					df.GalExt=[galex for i in 1:nrow(df)]
					df.Telescope=["GROND" for i in 1:nrow(df)]
					rename!(df,"Column1" => "Filter","Column2" => "Mag")
					if "Column3" in names(df); rename!(df, "Column3" => "MagErr"); end
                    
                    
                elseif occursin("RATIR", txt)
                	println(" RATIR report") 
             	    he=first(findfirst(r"^(( *)?)(g|r|i|Z|J|H|K)([^(\.)|(\w)])"m,txt))
			  	    ptxt=txt[1:he]
					t=match(r"(((\d+\.?)(\d{1,2})?)\s*(-?)(\b((minutes)|(mins?)|m\s)|((seconds?)|(sec)|s\s)|((hours?)|(hrs?))\b)?)\s+to\s+(((\d+\.?)(\d{1,2})?)( {1,2})?-?\s+(\b((minutes)|(mins?)|m\s)|((seconds?)|(sec)|s\s)|((hours?)|(hrs?))\b))\s+(after\s+the\s+(\w)+\s+trigger)",ptxt)
					t1=t.captures[2]
					t2=t.captures[17]
					ts=t.captures[6]
					te=t.captures[21]
					ft=tryparse(Float32,t1)
					st=tryparse(Float32,t2)
					if t.captures[28] != nothing
						ttwo=st*3600
					elseif t.captures[22] != nothing 
						ttwo=st*60
					else ttwo=st
					end
					if ts==nothing
						ts=te
					end
					if occursin(r"((hours?)|(hrs?))",ts)
						tone=ft*3600
					elseif occursin(r"((minutes)|(mins?)|m\s)",ts)
						tone=ft*60
					else tone=ft
					end
					midt=(ttwo+tone)/2
					e=last(collect(eachmatch(r"((\d+\.?)(\d{1,2})?)( {1,2})?-?(\b((minutes)|(mins?)|m\s)|((seconds?)|(sec)|s\s)|((hours?)|(hrs?))\b)",ptxt)))
					if e.captures[6] != nothing
						p=tryparse(Float16,e.captures[1]) *60
					elseif e.captures[9] != nothing
					    p=tryparse(Float16,e.captures[1]) 
					else p=tryparse(Float16,e.captures[1]) *3600
					end
            	    lr=first(findnext(r"^(((?:[\t ]*(?:\r?\n|\r))+)|(The)|(Given)|(This)|(These))"m,txt,he))-1
					ltxt=txt[lr:end]
					sys=match(r"((Vega)|(Johnson)|(AB)|(SDSS)|(DSS)|(2MASS)|(USNO)|(NOMAD)|(GSC)|(ST))",ltxt)
					syst=sys.captures[1]
					gal=match(r"(\s+are\s+(not)\s+corrected\s+for\s+Galactic\s+extinction\s+)",ltxt)
					gext=gal.captures[2]
					if gext==nothing
						galext="y"
					elseif gext=="not"
						galext="n"
					end
                	cltxt=replace(txt[he-1:lr], r"(mag)|(and)"=>"", r" ?(\+/?-) ?"=>",", r" +(>|=)? *"=>",",r"\n *"=>"\n",r"\t>?=? *"=>",",r" *�* *"=>"")
            		df=CSV.read(IOBuffer(cltxt), DataFrame, delim="," ,header=0)
					rename!(df,"Column1" => "Filter","Column2" => "Mag")
					if "Column3" in names(df); rename!(df, "Column3" => "MagErr"); end
					df.Time=[midt for i in 1:nrow(df)]
					df.Exp=[p for i in 1:nrow(df)]
					df.System=[syst for i in 1:nrow(df)]
					df.Telescope=["RATIR" for i in 1:nrow(df)]
					df.GalExt=[galext for i in 1:nrow(df)]
					
				elseif occursin("IKI", txt)
              	  println(" IKI report")                
              		he=first(findfirst(r"^(Date)|(UT start)|(T0+)"m,txt))
                	lr=first(findnext(r"^((The)|(Given)|(This)|(Photometry))"m,txt,he))-1
					ltxt=txt[lr:end]
					stm=r"(\b(Vega)|(Johnson)|(AB)|(SDSS)|(DSS)|(2MASS)|(USNO)|(NOMAD)|(GSC)|(ST)\W)"
					pho=collect(eachmatch(r"(photometry.*based.*stars?)",txt))
					if pho!=RegexMatch[]
						phot=(last(pho)).captures[1]
						ys=(collect(eachmatch(stm,phot)))
						sys=last(ys)
						syst=sys.captures[1]
						else syst="Check"
					end
					ga=match(r"(\s+are\s+(not)\s+corrected\s+for\s+((\w*)?(\s+)?)*?Galactic\s+(\w*)?(\s+)?(\3)?extinction\s+)",ltxt)
					if ga==nothing
						galex="n"
					elseif ga!=nothing
						gal=ga.captures[2]
						if gal==nothing
							galex="y"
						else galex="n"
						end
					end
                	cltxt=replace(txt[he:lr],"(mid)"=>"",r"UT ?start"i=>"UT_start","UL(3 sigma)"=>"Limit",r" ?(=|>|~|<)"=>"|" , r"\+/?-"=>"|",r"([� ]+)"=>"|")
                	df=CSV.read(IOBuffer(cltxt), DataFrame, skipto=4, delim="|"  ,ignorerepeated=true)
                	df.Telescope=["IKI" for i in 1:nrow(df)]
					df.System=[syst for i in 1:nrow(df)]
					df.GalExt=[galex for i in 1:nrow(df)]
					change(df)
					
				elseif occursin("KAIT", txt)
                    println(" KAIT report")                                
                    t=match(r"(((\d+\.?)(\d{1,5})?)( {1,})?-?\s+(\b((minutes)|(mins?)|m\s)|((seconds?)|(sec)|s\s)|((hours?)|(hrs?))|(days?)\b))(\s*after\s+(\w*\s+)?((burst)|(trigger)))",txt)
					t1=t.captures[2]
					ts=t.captures[6]
					ft=tryparse(Float32,t1)
					if t.captures[16] != nothing
						ttwo=ft*86400
					elseif t.captures[13] != nothing
						ttwo=ft*3600
					elseif t.captures[7] != nothing 
						ttwo=ft*60
					else ttwo=ft
					end
					mg=match(r"(.{50}\b((magnitude)|(mag))\b.{50})"si, txt)
					mgt=mg.captures[1]
					mgtx=match(r"((~|\s)(([0-2]?\d)(\.)[^:](\d{0,2}))\b)",mgt)
					magtxt=mgtx.captures[3]
             	    df=DataFrame()
					df.Mag=[magtxt for i in 1:1]
					df.System=["Vega" for i in 1:1]
					df.Telescope=["KAIT" for i in 1:1]
					df.Time=[ttwo for i in 1:1]
               	    
                end # if occursin
                dframe(df)
                select!(dfg, :GCN, :GRB, :Telescope, :Mag, :MagErr, :Time, :Exp, "Filter", :Limit,  :System, "      Date Time      ")       
            catch e
            if isa(e, LoadError); continue; end
            end # trycatch
            
        end # for loop
        if !isnothing(dfg)
        	CSV.write("dataall.csv",dfg)
        else
        @info "no dfg to write"
        end # !isnothing
end # function doanalysis
doanalysis()
