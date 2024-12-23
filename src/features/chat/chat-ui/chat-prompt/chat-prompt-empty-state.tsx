import { Button } from "@/components/ui/button";
import { FC, useState , FormEvent, useRef , Fragment ,useEffect} from "react";
import { Card } from "@/components/ui/card";
interface Props {}
import { useParams, useRouter } from "next/navigation";
import { useGlobalMessageContext } from "@/features/global-message/global-message-context";
interface Prop {}
import List from '@mui/material/List';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Collapse from '@mui/material/Collapse';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';
import { CheckIcon, ClipboardIcon, UserCircle } from "lucide-react";
import { useSession } from "next-auth/react";
import {AddPrompt,queryPrompt,markAsDeleted,updateItem} from "@/features/chat/chat-ui/chat-prompt/chat-prompt-cosmos";
import { PromptList } from "@/features/chat/chat-services/models";
import { Trash } from "lucide-react";
const ChatPromptEmptyState: FC<Props> = (props) => {
  const [open, setOpen] = useState(true);
  const [open_personal, setOpen_personal] = useState(true);
  const [promptTitle, setPromptTitle] = useState("");
  const [promptContent, setPromptContent] = useState("");
  const [dept, setDept] = useState("");
  const [promptId, setPromptId] = useState(0);
  //const ll = setDept("個人");
  //const [promptNew,setPromptNew] = useState(false);
  const titleChange = (value:string) => {
    //console.log(value);
    setPromptTitle(value);
  }
  const contentChange = (value:string) => {
    setPromptContent(value);
  }
  const { data: session } = useSession();
  const [prompt , setPrompt] = useState<Prompt[]>([]);
  const [promptDept , setPromptDept] = useState<PromptDept[]>([]);
/*  useEffect(() => {
    getPrompt();
  }, []); 
*/
// Run getPrompt only when session is available
/*useEffect(() => {
  if (session) {
    getPrompt();
  }
}, [session]); // Depend on session so it runs when session data is ready
*/
  type Prompt = {
    title: string;
    content: string;
    id: number;
    dept: string;
    username: string;
  };  
  type PromptDept = {
    title: string;
    content: string;
    id: number;
    dept: string;
    username: string;
  };
  const handleClick_company_all = () => {
    setOpen(!open);
  };
  const handleClick_personal_all = () => {
    setOpen_personal(!open_personal);
  };
  
  
  const listClick = (title:string,content:string,dept:string,id:number) => {
    const currentTimestamp = Date.now();
    const idNew = Number(currentTimestamp);
    if (id === 0) {
      //alert("新しいプロンプトを左のリストに追加してください。例：メール文書の要約");
      
      setPromptTitle("");
      setPromptContent("");
      setDept(dept);
      setPromptId(0);
      
    } else {
      setPromptTitle(title);
      setPromptContent(content);
      setDept(dept);
      setPromptId(id);
    }
  };  
  const listDelete = (id:string) => {
    if (window.confirm("削除しますか?")) {
      //setPrompt(prompt.filter((item) => item.id !== id));
      (async () => {
        try {
          const res = await markAsDeleted(id);
          //const list = await queryPrompt("個人",session?.user.name || "");
          //setPrompt([...list.map(item => ({ ...item, id: Number(item.id), username: "" }))]);          
          alert("削除しました");
        } catch (error) {
          console.error("エラーが発生しました:", error);
          alert("エラーが発生しました"+error);
        }
      })(); 
    }
  }
  const [isIconChecked, setIsIconChecked] = useState(false);
  const toggleIcon = () => {
    setIsIconChecked((prevState) => !prevState);
  };
  const handleButtonClick = () => {
    toggleIcon();
    navigator.clipboard.writeText(promptContent);
  }; 
  const saveButtonClick = () => {
    const currentTimestamp = Date.now();
    const id = Number(currentTimestamp);
    
    const newPrompt = { 
      title: promptTitle,
      content: promptContent,
      id: "0",
      dept: dept.trim() === "" ? "個人" : dept,
      usename: "",
      createdAt: new Date(),
      isDeleted: false
    };
    if (window.confirm("保存しますか?")) {
      if (promptId === 0) {
        //setPrompt([...prompt, { ...newPrompt, id: Number(newPrompt.id) }]);
        (async () => {
          try {
            newPrompt.id = id.toString();
            newPrompt.usename = session?.user.name || "";
            const res = await AddPrompt(newPrompt);
            //const list = await queryPrompt("個人",session?.user.name || "");
            //setPrompt([...prompt, ...list.map(item => ({ ...item, id: Number(item.id), username: "" }))]);          
            alert("保存しました");
            setPromptTitle("");
            setPromptContent("");
            //const sav = getPrompt();
          } catch (error) {
            console.error("エラーが発生しました:", error);
            alert("エラーが発生しました"+error);
          }
        })(); 
      } else {
        (async () => {
          try {
            const res = await updateItem(promptId.toString(), promptTitle, promptContent);
            //const list = await queryPrompt("個人",session?.user.name || "");
            //setPrompt([...prompt, ...list.map(item => ({ ...item, id: Number(item.id), username: "" }))]);          
            alert("保存しました");
          } catch (error) {
            console.error("エラーが発生しました:", error);
            alert("エラーが発生しました"+error);
          }
        })(); 
        
      }
    }
  }; 
 /* 
  const getPrompt = async () => {
    const list = await queryPrompt("個人",session?.user.name || "");
    setPrompt([...list.map(item => ({ ...item, id: Number(item.id), username: "" }))]);
    //alert("a")
  };
*/
  function PromtList(props:{dept?:string}): JSX.Element {
 
    const [prompt, setPrompt] = useState<any[]>([]);
    const { dept } = props;
  
    useEffect(() => {
      const fetchPromptList = async () => {
        // Fetch the prompt data, for example by calling queryPrompt
        const list = await queryPrompt("個人", session?.user.name || "");
        
        // Set the prompt state with the fetched list and ensure the ids are numbers
        setPrompt(list.map(item => ({ ...item, id: Number(item.id), username: "" })));
      };
  
      fetchPromptList();
    }, []);
  
    // Filter the prompt list based on the department if provided
    const newPrompt = dept ? prompt.filter((item) => item.dept === dept) : prompt;
  
    return (
      <>
        {newPrompt.map((item) => (
          <ListItemButton key={item.id} sx={{ pl: 4 }}>
            <ListItemText 
              secondary={item.title} 
              onClick={() => listClick(item.title, item.content, item.dept, item.id)} 
            />
            <Trash 
              size={16} 
              onClick={() => listDelete(item.id.toString())}  
            />
          </ListItemButton>
        ))}
      </>
    );
  }
  function PromtDept(props:{dept?:string}): JSX.Element {
 
    const [prompt, setPrompt] = useState<any[]>([]);
    const { dept } = props;
  
    useEffect(() => {
      const fetchPromptDept = async () => {
        // Fetch the prompt data, for example by calling queryPrompt
        const list = await queryPrompt("会社全体", session?.user.name || "");
        
        // Set the prompt state with the fetched list and ensure the ids are numbers
        setPrompt(list.map(item => ({ ...item, id: Number(item.id), username: "" })));
      };
  
      fetchPromptDept();
    }, []);
  
    // Filter the prompt list based on the department if provided
    const newPrompt = dept ? prompt.filter((item) => item.dept === dept) : prompt;
  
    return (
      <>
        {newPrompt.map((item) => (
          <ListItemButton key={item.id} sx={{ pl: 4 }}>
            <ListItemText 
              secondary={item.title} 
              onClick={() => listClick(item.title, item.content, item.dept, item.id)} 
            />
            <Trash 
              size={16} 
              onClick={() => listDelete(item.id.toString())}  
            />
          </ListItemButton>
        ))}
      </>
    );
  }
  return (
    <div className="grid grid-cols-7 h-full w-full items-center container mx-auto max-w-4xl justify-center h-full gap-1">
      <Card className="col-span-3 flex flex-col gap-1 p-o h-full w-full">
      <p className="text-xs text-muted-foreground">
        <div className="col-span-2 gap-1 flex flex-col flex-1 justify-start text-xs"> 
        <List 
          //sx={{ width: '100%', maxWidth: 360, bgcolor: 'background.paper' }}
          className="min-h-fit bg-background shadow-sm resize-none py-1 w-full"
          component="nav" 
        >
          <ListItemButton　onClick={handleClick_company_all}>
            <ListItemText className="text-xs" secondary="会社全体" />
            {open ? <ExpandLess /> : <ExpandMore />}
          </ListItemButton> 
          <Collapse in={open} timeout="auto" unmountOnExit>
          {session?.user?.isAdmin ? (
            <ListItemButton sx={{ pl: 4 }}>
              <ListItemText secondary="新規" onClick={() => listClick("", "","会社全体",0)} />
            </ListItemButton>
        ) : (
          <></>
        )}
          <List component="div" disablePadding>
            <PromtDept/>
           </List>            
          </Collapse>     
          <ListItemButton onClick={handleClick_personal_all}>
            <ListItemText secondary="個人" />
            {open_personal ? <ExpandLess /> : <ExpandMore />}
          </ListItemButton>
          <Collapse in={open_personal} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              <ListItemButton sx={{ pl: 4 }}>
                <ListItemText secondary="新規" onClick={() => listClick("", "","個人",0)} />
              </ListItemButton>
              <PromtList dept="個人"/>
            </List>
          </Collapse>                                
         </List>  
        </div>
      </p>
      </Card>
     <Card className="col-span-4 flex flex-col gap-1 p-5 h-full w-full">
          <p className="text-xs text-muted-foreground">
          <label>タイトル</label>
          <div className="flex gap-3 items-center flex-1 p-0">
            <textarea
              name = "title"
              className="min-h-fit bg-background shadow-sm resize-none py-1 w-full"
              value={promptTitle==="" ? "" : promptTitle}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => titleChange(e.target.value)}
            ></textarea>
          <Button
            variant={"ghost"}
            size={"sm"}
            title="Copy text"
            className="justify-right flex"
            onClick={handleButtonClick}
          >
            {isIconChecked ? (
              <CheckIcon size={16} />
            ) : (
              <ClipboardIcon size={16} />
            )}
          </Button>            
          </div>
          </p>
          <p className="text-xs text-muted-foreground">
            <input type = "hidden" name = "dept" value = {dept}></input>
            <input type = "hidden" name = "id" value = {promptId}></input>
          </p>
          <p className="text-xs text-muted-foreground">
          <label>プロンプト</label>
          <textarea
              name = "prompt"
              className="min-h-fit w-full bg-background shadow-sm resize-none py-4 h-[60vh]"
              value={promptContent==="" ? "" : promptContent}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => contentChange(e.target.value)}
            >{promptContent==="" ? "" : promptContent}</textarea>
            </p>
          <Button
            variant={"ghost"}
            size={"sm"}
            title="保存"
            className="justify-right flex bg-green-500 text-white"
            onClick={() => saveButtonClick()}
                     >
            保存
          </Button>
         </Card>
    </div>
  );
};
export default ChatPromptEmptyState;