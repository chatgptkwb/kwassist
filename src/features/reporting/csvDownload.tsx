"use client"
import { Button } from "@/components/ui/button";
import {useRef} from "react";
import { CSVLink,CSVDownload } from "react-csv" ;
import { SqlQuerySpec } from "@azure/cosmos";
import ChatRow from "@/components/chat/chat-row";
import { useState } from "react";

//フェッチの完了を判断するステート(完了後に true とする)
import { FindAllChatThreadsForReporting } from "./reporting-service";

export type ReportingProp = {
  searchParams: {
    pageSize?: number;
    pageNumber?: number;
  };
};

//const [fetchDone, setFetchDone] = useState(false);


import {
  CHAT_THREAD_ATTRIBUTE,
  ChatMessageModel,
  ChatThreadModel,
  MESSAGE_ATTRIBUTE,
} from "../chat/chat-services/models";
import { CosmosDBContainer } from "../common/cosmos";

const headers = [
  { label: "会話日時", key: "createdAt" },
  { label: "ユーザー名", key: "useName" },
  { label: "タイトル", key: "name" },
  { label: "スレッドID", key: "id" }
];
 

//  function Download () {
  const Download :  React.FC<Props> = (props) => {
    const data = props.resources;
      return  <CSVLink data={data} headers={headers}><Button >CSVダウンロード</Button></CSVLink>
      //return  <CSVLink data={data} ><Button >CSVダウンロード!</Button></CSVLink>
    

    // return
}   
  
interface Props {
  resources: Array<ChatThreadModel>;
}

  const  DownloadCSV :  React.FC<Props> = (props) => {

    return (
      <main>
        <div>
        <Download {...props} />
        </div>
        <div>
      </div>
      </main>
    );
  }
export default DownloadCSV;
