
use std::{os::macos::raw::stat, sync::Arc};

use log::{info,  warn};
use serde::Serialize;
use tokio::{sync::mpsc::{self, UnboundedSender}, task::JoinHandle};


#[derive(Serialize, Copy, Clone, PartialEq, Debug)]
pub enum ServerStatus {
    None,
    Starting,
    Stopped,
    Running,
    // Unhealthy, // TODO: find out what unhealthy state is and make a way to set it
}

pub struct HealthService {
    status: Arc<ServerStatus>,
    handle: Option<JoinHandle<()>>,
    tx: UnboundedSender<ServerStatus>,
}

impl HealthService{
    pub fn start(status: ServerStatus) -> Self{
        let status = Arc::new(status);
        let (tx, mut rx) = mpsc::unbounded_channel::<ServerStatus>();
        let handle = tokio::spawn(async move {
            info!("HealthService background task started");
            while let Some(status) = rx.recv().await {
                
            }

            rx.close();
            warn!("HealthService background task stopped");
        });
        Self { tx, handle: Some(handle), status}
    }
}